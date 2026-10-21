
import os
import random
import json
import unicodedata
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from CompleteBin.IO import writePickle
from CompleteBin.Trainer.optimized_debiased_loss import debiased_info_nce_loss_optimized


from CompleteBin.logger import get_logger
from CompleteBin.Trainer.debiased_loss import build_same_genome_mask
from CompleteBin.Trainer.pair_head import pair_loss
from CompleteBin.Trainer.warmup import build_schedule
from CompleteBin.Trainer.pretrain_contrastive_loss import (
    genome_contrastive_top1_accuracy,
    genome_supervised_contrastive_loss,
)

logger = get_logger()


# bsz : batch size (number of positive pairs)
# d : latent dim
# x : Tensor, shape=[bsz, d]
# latents for one side of positive pairs
# y : Tensor, shape=[bsz, d]
# latents for the other side of positive pairs

# Alignment & Uniformity, the lower the better
def align_loss(x, y, alpha=1):
    return (x - y).norm(dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def min_max_norm(input_tensor: torch.Tensor):
    max_vals, _ = torch.max(input_tensor, dim=-1, keepdim=True)
    min_vals, _ = torch.min(input_tensor, dim=-1, keepdim=True)
    dis = max_vals - min_vals
    # print(dis.shape)
    return (input_tensor - min_vals) / dis


# ── 预训练指标表 ──────────────────────────────────────────────────────────
# test_metrics.md 由 test_metrics.jsonl 渲染而来：每个 epoch 重写整张表，列宽按
# 该列最长内容补空格，所以在纯文本里也能对齐（不是只有第一列对齐的那种追加写法）。
# 单独的渲染脚本 scripts/pretrain_run_20260919/render_metrics_md.py 复用同一套定义。

PRETRAIN_METRIC_COLUMNS = (
    "Epoch", "Selection score", "Test short R@1", "768–1500 R@1",
    "1500–2500 R@1", "Overall R@1", "Overall R@5", "Pair AUROC",
    "Pair AUPRC", "Train-ref top-1", "Train-ref top-5", "Train-ref CE",
    "Train SupCon", "Train SupCon Acc1", "Train CE", "Train Pair",
    "Train Pair Acc",
)


def _display_width(text: str) -> int:
    """终端里汉字占两格，补空格时按这个宽度算，不能用 len()。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in text)


def _format_metric(value, fmt: str = "{:.6f}") -> str:
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return "nan"


def _format_percent(value) -> str:
    """jsonl 里存的是 0–1 的比例，表里统一显示成百分比。"""
    try:
        return f"{float(value) * 100:.2f}"
    except (TypeError, ValueError):
        return "nan"


def _cell(value) -> str:
    """单元格文字：数值走 _format_metric；字符串（如 "-" 表示该口径不可得）原样保留。"""
    return value if isinstance(value, str) else _format_metric(value)


def _cell_percent(value) -> str:
    return value if isinstance(value, str) else _format_percent(value)


def pretrain_metric_cells(metrics: dict) -> list:
    """一行指标（一个 epoch）→ 表格各列的文字。键缺失时填 nan，方便旧日志复用。

    带 ``label`` 的行（比如"上游基线"）用 label 当第一列，排在所有 epoch 后面。
    字符串值（比如 "-"）原样输出，用于标明该列口径不可得。
    """
    test = metrics.get("test") or {}
    train = metrics.get("train") or {}
    ref = metrics.get("train_reference") or {}
    if metrics.get("label"):
        epoch = str(metrics["label"])
    else:
        try:
            epoch = f"{int(metrics['epoch'])}"
        except (KeyError, TypeError, ValueError):
            epoch = "nan"
    # Selection score 一律按当前公式(0.3×短片段检索 + 0.7×train-ref)从分项重算，
    # 这样改过公式之后，表里所有行（含早期用旧公式算的行和基线行）都是同一口径。
    short_r1 = test.get("short_macro_recall_at_1")
    ref_top1 = ref.get("top1_accuracy")
    if isinstance(short_r1, (int, float)) and isinstance(ref_top1, (int, float)):
        selection = 0.3 * float(short_r1) + 0.7 * float(ref_top1)
    else:
        selection = metrics.get("selection_score")
    return [
        epoch,
        _cell(selection),
        _cell(test.get("short_macro_recall_at_1")),
        _cell(test.get("length_768_1500_macro_recall_at_1")),
        _cell(test.get("length_1500_2500_macro_recall_at_1")),
        _cell(test.get("overall_recall_at_1")),
        _cell(test.get("overall_recall_at_5")),
        _cell(test.get("pair_auroc")),
        _cell(test.get("pair_auprc")),
        _cell(ref.get("top1_accuracy")),
        _cell(ref.get("top5_accuracy")),
        _cell(ref.get("classification_loss")),
        _cell(train.get("supcon")),
        _cell_percent(train.get("supcon_acc1")),
        _cell(train.get("classification")),
        _cell(train.get("pair")),
        _cell_percent(train.get("pair_accuracy")),
    ]


def render_pretrain_metrics_markdown(jsonl_path: str, markdown_path: str,
                                     extra_rows_path: str = None) -> int:
    """把 test_metrics.jsonl 渲染成对齐的 markdown 表，返回写出的行数。

    如果同目录下有 ``baseline_metrics.json``（或显式传入 extra_rows_path），里面的
    行会追加到表尾——用来放不随 epoch 变化的参照项，比如上游 CompleteBin 的权重。
    这些行不需要 ``epoch``，但要有 ``label``（会显示在第一列）。
    """
    rows = []
    has_extra = False
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    rows.append(pretrain_metric_cells(record))
    extra_rows_path = extra_rows_path or os.path.join(
        os.path.dirname(os.path.abspath(markdown_path)), "baseline_metrics.json")
    if os.path.exists(extra_rows_path):
        try:
            with open(extra_rows_path, "r", encoding="utf-8") as handle:
                extra = json.load(handle)
        except json.JSONDecodeError:
            extra = None
        for record in (extra if isinstance(extra, list) else [extra] if extra else []):
            if isinstance(record, dict):
                rows.append(pretrain_metric_cells(record))
                has_extra = True
    # jsonl 是追加写的：数字 epoch 按大小排序；带 label 的附加行（上游基线）
    # 排在最前面，方便一眼看到参照值
    rows.sort(key=lambda cells: (0, 0) if not cells[0].isdigit()
              else (1, int(cells[0])))

    widths = []
    for index, title in enumerate(PRETRAIN_METRIC_COLUMNS):
        width = _display_width(title)
        for cells in rows:
            width = max(width, _display_width(cells[index]))
        widths.append(width)

    lines = ["# Pretrain metrics", ""]
    lines.append("| " + " | ".join(
        title.rjust(widths[index] - _display_width(title) + len(title))
        for index, title in enumerate(PRETRAIN_METRIC_COLUMNS)) + " |")
    lines.append("|" + "|".join("-" * (width - 1) + ":" for width in widths) + "|")
    for cells in rows:
        padded = [
            cell.rjust(width - _display_width(cell) + len(cell))
            for cell, width in zip(cells, widths)
        ]
        lines.append("| " + " | ".join(padded) + " |")
    if has_extra:
        lines += [
            "",
            "注：`上游基线` 那一行是把 "
            "`/home/ericluzhang/ZouBohao/CompleteBin-DB-v1.2.3.0/CheckPoint/"
            "pretrain_weight_large_hidden_dim_768_layers_4.pth`（4 层 / hidden 768 / "
            "分类头 15434 类）用**同一套评测**跑出来的：检索各项与训练循环里用的是"
            "同一个 `test_retrieval`；它的类别编号不可得（它的参考库有 15433 个文件，"
            "本机没有那份清单），Train-ref top-1 用「样本池投票定类」的代理口径"
            "（在我们自己的 checkpoint 上校准过，误差约 +0.005），top-5 与 CE 因此留空。",
        ]

    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(rows)


def schedule_of_temperature(temp: float, epochs: int):
    start_temp = temp - 0.005
    end_temp = temp + 0.005
    step = 0.01 / epochs
    res = [0 for _ in range(epochs)]
    for i, cur_temp in enumerate(np.arange(start_temp, end_temp, step)):
        if i <= epochs - 1:
            res[i] = float("%.3f" % cur_temp)
    res[-1] = end_temp
    for i, ite in enumerate(res):
        if ite == 0:
            res[i] = temp
    return res


class Trainer(object):

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 device: str,
                 epochs: int,
                 model_save_folder: str,
                 n_views: int,
                 batch_size: int,
                 dropout: float,
                 max_cov_mean,
                 temperature_simclr=0.123,
                 log_every_n_steps: int = 10,
                 seq_length=7,
                 use_pretrained: bool = True,
                 ):
        self.device = device
        self.dropout = dropout
        self.model = model.to(device)
        self.optimizer = optimizer
        self.seq_length = seq_length
        # self.optimizer2 = optimizer2
        self.scheduler = scheduler
        self.epochs = epochs
        self.model_save_folder = model_save_folder
        self.log_every_n_steps = log_every_n_steps
        self.n_views = n_views
        self.batch_size = batch_size
        # self.max_cov_mean = torch.tensor(max_cov_mean[0], dtype=torch.float32)[None, :, None].to(self.device)
        self.max_cov_mean_1 = torch.tensor(max_cov_mean[1], dtype=torch.float32)[None, None, :].to(self.device)  # 1, 1, bam_files
        self.max_cov_mean_2 = torch.tensor(max_cov_mean[0], dtype=torch.float32)[None, :, None].to(self.device)  # 1, bam_files, 1
        self.max_cov_std = torch.tensor(max_cov_mean[2], dtype=torch.float32)[None, None, :].to(self.device)  # 1, 1, bam_files
        logger.info(f"--> The max of coverage mean value of whole contig 1 is {self.max_cov_mean_1} shape {self.max_cov_mean_1.shape}..")
        logger.info(f"--> The max of coverage mean value of whole contig 2 is {self.max_cov_mean_2} shape {self.max_cov_mean_2.shape}..")
        logger.info(f"--> The max of coverage std value of whole contig is {self.max_cov_std} shape {self.max_cov_std.shape}..")
        self.temperature_simclr = temperature_simclr
        self.temperature_schedule = schedule_of_temperature(temperature_simclr, epochs)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.scaler = torch.amp.GradScaler()
        self.use_pretrained = use_pretrained

    def get_model_inputs(self, n_views_tuple_list):
        seq_rad_tokens_n_views = []
        mean_tokens_n_views = []
        std_tokens_n_views = []
        whole_bp_cov_tnf_array_n_views = []
        for i, (cur_seq_rad_tokens, cur_mean_tokens,
                cur_std_tokens, cur_whole_bp_cov_tnf_array) in enumerate(n_views_tuple_list):
            seq_rad_tokens_n_views.append(cur_seq_rad_tokens)  # [bs, views]
            mean_tokens_n_views.append(cur_mean_tokens)
            std_tokens_n_views.append(cur_std_tokens)
            whole_bp_cov_tnf_array_n_views.append(cur_whole_bp_cov_tnf_array)
        seq_rad_tokens_n_views = torch.cat(seq_rad_tokens_n_views, dim=0).to(torch.float32).to(self.device, non_blocking=True)
        mean_tokens_n_views = torch.cat(mean_tokens_n_views, dim=0).to(torch.float32).to(self.device, non_blocking=True)
        std_tokens_n_views = torch.cat(std_tokens_n_views, dim=0).to(torch.float32).to(self.device, non_blocking=True)
        whole_bp_cov_tnf_inputs = torch.cat(whole_bp_cov_tnf_array_n_views, dim=0).to(torch.float32).to(self.device, non_blocking=True)
        ##################
        ## Changed Here ##
        ##################
        # mean_tokens_n_views = torch.log(mean_tokens_n_views + 1.)
        # std_tokens_n_views = torch.log(std_tokens_n_views + 1.)
        # whole_bp_cov_tnf_inputs = torch.log(whole_bp_cov_tnf_inputs + 1.)

        mean_tokens_n_views /= self.max_cov_mean_1  # n*b, l, bam_files
        std_tokens_n_views /= self.max_cov_std  # n*b, l, bam_files
        whole_bp_cov_tnf_inputs /= self.max_cov_mean_2  # b, bam_files, dim_L

        assert len(mean_tokens_n_views.shape) == 3 and len(std_tokens_n_views.shape) == 3, \
            ValueError(
                f"The dim mean_input is {mean_tokens_n_views.shape}, The dim std_inputs is {std_tokens_n_views.shape}. One of them not equal with 2.")
        return (seq_rad_tokens_n_views,
                mean_tokens_n_views, std_tokens_n_views, whole_bp_cov_tnf_inputs)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader = None, model_weight_path=None):
        logger.info(f"--> Start self-supervised training with {self.epochs} epochs.")
        logger.info(f"--> Training with {self.device} device. info_nce_loss_for_loop")
        loss_record = {}
        if model_weight_path is not None:
            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
            logger.info(f"--> Model weight has been loaded.")
        for epoch_counter in range(1, self.epochs + 1):
            n_iter = 0.
            total_iters = len(train_loader)
            self.model.train()
            for n_views_tuple_list in tqdm(train_loader):
                n_views = len(n_views_tuple_list)
                assert n_views == self.n_views, ValueError(f"Views number is not equal with each other. {n_views}, {self.n_views}")
                (seq_rad_tokens_n_views,
                 mean_tokens_n_views, std_tokens_n_views,
                 whole_bp_cov_tnf_inputs) = \
                    self.get_model_inputs(n_views_tuple_list)
                # ============ multi-res forward passes ... ============
                self.optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    simclr_emb_contrast, _, seq_taxon_enc = self.model.forward(
                        seq_rad_tokens_n_views,
                        mean_tokens_n_views,
                        std_tokens_n_views,
                        whole_bp_cov_tnf_inputs,
                        n_views)

                    sg_mask = build_same_genome_mask(
                        seq_taxon_enc[:self.batch_size],
                        # 传 classify（分类头的完整前向）而不是裸 out_linear，
                        # 否则将来分类头挂了前置层时，这里的 logits 会和别处不一致
                        self.model.pretrain_model.classify if self.use_pretrained else None,
                        epoch=epoch_counter,
                        n_iter=int(n_iter),
                        total_iters=total_iters,
                        enable_mask=self.use_pretrained,
                    )

                    loss_simclr, logits, labels = debiased_info_nce_loss_optimized(
                        simclr_emb_contrast,
                        self.batch_size,
                        self.n_views,
                        self.temperature_schedule[epoch_counter - 1],
                        self.device, self.criterion, sg_mask,
                    )
                    loss = loss_simclr

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # logger
                if n_iter % self.log_every_n_steps == 0:
                    acc1 = accuracy(logits, labels)[0]
                    active_mask = sg_mask.sum() / max(self.batch_size * self.batch_size - self.batch_size, 1)
                    logger.info(
                        f"-->Epoch:{epoch_counter}/{self.epochs}"
                        f"|LossSum:{loss.item():.2f}"
                        f"|LossSimCLR:{loss_simclr.item():.2f}"
                        f"|NViewsAcc1:{acc1.item():.2f}"
                        f"|Mask:{active_mask:.6f}"
                        f"|LR:{self.optimizer.param_groups[0]['lr']:.8f}"
                        f"|Temp:{self.temperature_schedule[epoch_counter - 1]}"
                    )
                n_iter += 1
            self.scheduler.step()
            self.model.eval()
            losses_valid = self.valid(valid_loader)
            self.model.train()
            loss_record[epoch_counter] = losses_valid
            # save model checkpoints
            checkpoint_name = 'checkpoint_{}.pth'.format(epoch_counter)
            torch.save(self.model.state_dict(), os.path.join(self.model_save_folder, checkpoint_name))
            logger.info(f"--> Model checkpoint has been saved at {os.path.join(self.model_save_folder, checkpoint_name)}.")
        logger.info("--> Training has finished.")
        return loss_record

    def valid(self, valid_loader: DataLoader = None):
        logger.info(f"--> Start self-supervised validation.")
        logger.info(f"--> Valid with {self.device} device.")
        self.model.eval()
        loss = 0.
        align_loss_f = 0.
        uni_loss_f = 0.
        n_iter = 0.
        combined_x_o = []
        combined_x_p1 = []
        combined_x_p2 = []
        i = 1
        with torch.no_grad():
            for n_views_tuple_list, _ in tqdm(valid_loader):
                n_views = len(n_views_tuple_list)
                (seq_rad_tokens_n_views,
                 mean_tokens_n_views, std_tokens_n_views,
                 whole_bp_cov_tnf_inputs) = \
                    self.get_model_inputs(n_views_tuple_list)
                # ============ multi-res forward passes ... ============
                simclr_emb_contrast, _, _ = self.model.forward(seq_rad_tokens_n_views,
                                                               mean_tokens_n_views, std_tokens_n_views, whole_bp_cov_tnf_inputs, n_views)
                # Take only token 0 (fused k-mer+coverage): first n_views*bs rows
                # Model output layout: [n_views * seq_length * bs, feature_dim]
                #   [: n_views*bs]        = token 0 (fused k-mer+coverage)
                #   [n_views*bs:2*...]    = token 1 (pure_cov)
                #   [2*n_views*bs:]       = token 2 (g_kmer)
                token0 = simclr_emb_contrast[: n_views * self.batch_size]
                token0 = token0.view(n_views, self.batch_size, -1)
                x_o = token0[0]   # ori_view token 0
                x_p1 = token0[1]  # generated_view token 0
                x_p2 = token0[2]  # ori_view token 0 (same as x_o)
                combined_x_o.append(x_o)
                combined_x_p1.append(x_p1)
                combined_x_p2.append(x_p2)
                if i % 10 == 0:
                    combined_x_o = torch.cat(combined_x_o, dim=0)
                    combined_x_p1 = torch.cat(combined_x_p1, dim=0)
                    combined_x_p2 = torch.cat(combined_x_p2, dim=0)
                    loss_align = (align_loss(combined_x_o, combined_x_p1) + align_loss(combined_x_o, combined_x_p2) +
                                  align_loss(combined_x_p1, combined_x_p2)) / 3.
                    align_loss_f += loss_align
                    # print("loss_align", loss_align)
                    # uniform loss
                    loss_uni = (uniform_loss(combined_x_o) + uniform_loss(combined_x_p1) + uniform_loss(combined_x_p2)) / 3.
                    # print("loss_uni", loss_uni)
                    uni_loss_f += loss_uni
                    cur_loss = loss_align + loss_uni
                    loss += cur_loss
                    n_iter += 1.
                    combined_x_o = []
                    combined_x_p1 = []
                    combined_x_p2 = []
                i += 1
        if len(combined_x_o) != 0:
            combined_x_o = torch.cat(combined_x_o, dim=0)
            combined_x_p1 = torch.cat(combined_x_p1, dim=0)
            combined_x_p2 = torch.cat(combined_x_p2, dim=0)
            loss_align = (align_loss(combined_x_o, combined_x_p1) + align_loss(combined_x_o, combined_x_p2) +
                          align_loss(combined_x_p1, combined_x_p2)) / 3.
            align_loss_f += loss_align
            # uniform loss
            loss_uni = (uniform_loss(combined_x_o) + uniform_loss(combined_x_p1) + uniform_loss(combined_x_p2)) / 3.
            uni_loss_f += loss_uni
            cur_loss = loss_align + loss_uni
            loss += cur_loss
            n_iter += 1.
            combined_x_o = []
            combined_x_p1 = []
            combined_x_p2 = []
        self.model.train()
        logger.info(f"--> align loss is {align_loss_f / n_iter}, unif loss is {uni_loss_f / n_iter}")
        logger.info(f"--> The validation loss is {loss / n_iter}. Validation has finished.")
        return loss / n_iter

    def inference(self, infer_loader: DataLoader, output_folder_path: str, model_weight_path: str = None):
        logger.info(f"--> Inference with {self.device} device.")
        if model_weight_path is not None:
            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
        self.model.eval()
        self.optimizer.zero_grad()
        n_iter = 0
        simclr_contigname2emb_norm_ndarray = {}
        with torch.no_grad():
            for n_views_tuple_list, contigname_file_list in tqdm(infer_loader):
                (seq_rad_tokens_n_views,
                 mean_tokens_n_views, std_tokens_n_views,
                 whole_bp_cov_tnf_inputs) = \
                    self.get_model_inputs(n_views_tuple_list)
                # ============ multi-res forward passes ... ============
                simclr_emb, bs, _ = self.model.forward(seq_rad_tokens_n_views,
                                                       mean_tokens_n_views, std_tokens_n_views, whole_bp_cov_tnf_inputs, 1)
                simclr_emb = simclr_emb[0: bs]
                for i in range(len(contigname_file_list)):
                    prefix, suffix = os.path.splitext(contigname_file_list[i])
                    if suffix == ".pkl":
                        contigname = ">" + prefix
                    else:
                        contigname = ">" + contigname_file_list[i]
                    simclr_contigname2emb_norm_ndarray[contigname] = simclr_emb[i].detach().to("cpu").numpy()
                n_iter += 1
        writePickle(os.path.join(output_folder_path, f"SimCLR_contigname2emb_norm_ndarray.pkl"), simclr_contigname2emb_norm_ndarray)


class PretrainTrainer(object):

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 device: str,
                 epochs: int,
                 model_save_folder: str,
                 batch_size: int,
                 log_every_n_steps: int = 20,
                 contrastive_temperature: float = 0.10,
                 contrastive_loss_weight: float = 1.0,
                 classification_loss_weight: float = 0.10,
                 train_genome_names=None,
                 checkpoint_metadata=None,
                 pair_head=None,
                 pair_head_weight: float = 0.0,
                 pair_head_pairs: int = 4096,
                 ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.model_save_folder = model_save_folder
        self.log_every_n_steps = log_every_n_steps
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.0).to(self.device)
        self.contrastive_temperature = contrastive_temperature
        # 两路损失的配比：分类（CE）默认是主损失，SupCon 由调用方给系数。
        self.contrastive_loss_weight = contrastive_loss_weight
        self.classification_loss_weight = classification_loss_weight
        # 分类头和配对头都由外部传进来的同一个优化器更新（和主干共用 lr、
        # weight decay 和学习率曲线），这里不再持有任何 head 专用优化器。
        # Same-genome pair head, trained during the joint phase.
        self.pair_head = pair_head
        self.pair_head_weight = float(pair_head_weight)
        self.pair_head_pairs = int(pair_head_pairs)
        self.train_genome_names = train_genome_names or []
        self.checkpoint_metadata = checkpoint_metadata or {}
        os.makedirs(self.model_save_folder, exist_ok=True)

    def train(self, train_loader: DataLoader, test_loader: DataLoader,
              train_reference_loader: DataLoader, model_weight_path=None,
              max_epochs_this_run=None,
              resume_backbone_lr: float = None, resume_head_lr: float = None):
        logger.info(f"--> Start genome-contrastive pretraining with {self.epochs} epochs.")
        logger.info(f"--> Training with {self.device} device.")
        loss_record = {}
        best_key = None
        start_epoch = 0
        if model_weight_path is not None:
            # weights_only=False 是必须的：这个文件里除了模型权重还存了优化器和
            # 学习率调度器对象，PyTorch 2.6 起 torch.load 默认 weights_only=True
            # 会直接拒绝反序列化它们（报 Unsupported global: CosineAnnealingWarmRestarts）。
            loaded = torch.load(model_weight_path, map_location=self.device,
                                weights_only=False)
            state_dict = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
            self.model.load_state_dict(state_dict)
            if isinstance(loaded, dict) and "optimizer_state_dict" in loaded:
                try:
                    self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
                except (ValueError, KeyError) as error:
                    # Expected when resuming a checkpoint from a recipe with a
                    # different parameter-group layout (e.g. a frozen logit
                    # scale): keep the fresh optimiser instead of crashing.
                    logger.warning(
                        f"--> Optimizer state not restored ({error}); "
                        "continuing with a freshly initialised optimiser."
                    )
            resume_base_lrs = None
            if resume_backbone_lr is not None or resume_head_lr is not None:
                # 续跑时显式覆盖学习率（做 lr 扫描用）。约定：optimizer 的**最后一组**
                # 是分类头自己的组（pretrain.py 就是这么排的），其余组属于主干。
                # 注意这里改的是**基准** lr（曲线的起点），曲线本身还是要按剩余
                # epoch 数重新铺开——所以先把基准值算出来，再交给调度器。
                head_index = len(self.optimizer.param_groups) - 1
                resume_base_lrs = []
                for index, group in enumerate(self.optimizer.param_groups):
                    base = float(group["lr"])
                    if index == head_index:
                        if resume_head_lr is not None:
                            base = float(resume_head_lr)
                    elif resume_backbone_lr is not None:
                        base = float(resume_backbone_lr)
                    resume_base_lrs.append(base)
                    group["lr"] = base
                logger.info(
                    f"--> Resumed learning rates overridden: backbone="
                    f"{resume_backbone_lr}, head={resume_head_lr}."
                )
            # 学习率调度器**不恢复 state_dict**（旧版本是这么干的，见下），而是按
            # checkpoint 里存的纯数字 spec + 已完成的 epoch 数重建同一条曲线：
            #   · spec 里带着 base_lrs，所以重建出来的是"原本那条曲线"，不会因为
            #     checkpoint 存的是曲线中途的 lr 而把基准值弄错；
            #   · 曲线是 epoch 的纯函数，重建后紧接着的下一个 epoch 的 lr 与不中断
            #     跑完全一致。
            # 旧版本的失败模式：把 pickle 出来的调度器对象整体塞回 __dict__，里面
            # 持有的是反序列化出来的**优化器副本**，之后 step() 全在改副本，真实
            # 优化器的 lr 从此冻结（实测主干卡在 6.39e-4、头 3.995e-2 共 70 个 epoch）。
            if isinstance(loaded, dict):
                saved_metadata = loaded.get("metadata") or {}
                saved_formula = saved_metadata.get("selection_formula")
                current_formula = self.checkpoint_metadata.get("selection_formula")
                start_epoch = int(loaded.get("epoch", 0))
                saved_best_key = loaded.get("best_key")
                if saved_best_key is not None:
                    if saved_formula == current_formula:
                        best_key = tuple(saved_best_key)
                    else:
                        # 选择分数的口径变了（或旧 checkpoint 里根本没记录口径），
                        # 旧的最好分数和新分数不可比 → 不继承，从下一轮重新选
                        logger.warning(
                            "--> best_key not inherited: the checkpoint used "
                            f"'{saved_formula or 'no recorded formula'}' but this run "
                            f"uses '{current_formula}'; best will be re-selected from "
                            "the coming epochs."
                        )
            logger.info(f"--> Training state loaded; resuming after epoch {start_epoch}.")
            saved_schedule = (loaded.get("metadata") or {}).get("schedule") \
                if isinstance(loaded, dict) else None
            if resume_base_lrs is not None:
                saved_schedule = None  # 用了扫描用的覆盖值 → 基准值以覆盖值为准
            self.scheduler = build_schedule(
                self.optimizer,
                spec=saved_schedule,
                total_epochs=self.epochs,
                completed_epochs=start_epoch,
            )
            if resume_base_lrs is not None:
                self.scheduler.base_lrs = list(resume_base_lrs)
                self.scheduler._apply(start_epoch)
            logger.info(
                f"--> LR schedule rebuilt from spec (no scheduler state_dict restore): "
                f"total_epochs={self.scheduler.total_epochs}, "
                f"warmup_epochs={self.scheduler.warmup_epochs}, "
                f"base_lrs={[round(v, 8) for v in self.scheduler.base_lrs]}, "
                f"resume_at_epoch={start_epoch}, "
                f"lr_for_next_epoch={[round(v, 8) for v in self.scheduler.get_last_lr()]}, "
                f"lr_at_last_epoch={[round(base * self.scheduler.factor(self.epochs - 1), 8) for base in self.scheduler.base_lrs]}."
            )
        # 分类头现在是裸线性层，没有温度/logit scale/归一化需要重钉。
        end_epoch = self.epochs
        if max_epochs_this_run is not None:
            end_epoch = min(self.epochs, start_epoch + max_epochs_this_run)
        logger.info(f"--> This run covers epochs {start_epoch + 1}--{end_epoch}.")
        for epoch_counter in range(start_epoch + 1, end_epoch + 1):
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch_counter - 1)
            n_iter = 0
            self.model.train()
            running = defaultdict(float)
            # SupCon 的 top-1 命中率只在打日志的那些步算一次（每次要多算一遍
            # 两两余弦），所以单独累计、按计算次数取平均，不能并进 running。
            supcon_acc1_sum = 0.0
            supcon_acc1_count = 0
            length_bin_counts = [0, 0, 0, 0, 0]
            pair_type_counts = defaultdict(int)
            for batch in tqdm(train_loader):
                n_views = (len(batch) - 2) // 2
                views = [view.to(torch.float32).to(self.device, non_blocking=True)
                         for view in batch[:n_views]]
                batch_taxon_labels = batch[n_views].to(self.device, non_blocking=True).long()
                bins = batch[n_views + 1:n_views + 1 + n_views]
                pair_types = batch[-1]
                batch_genome_seq_tokens = torch.cat(views, dim=0)
                all_labels = batch_taxon_labels.repeat(n_views)
                self.optimizer.zero_grad()
                embeddings = self.model.encode(batch_genome_seq_tokens)
                projections = self.model.project_pretrain(embeddings)
                # 分类损失每个基因组只用一条片段，视图窗口逐步轮换。把同一基因组的
                # 多条近似重复片段放进同一个交叉熵 batch 会让分类头学不动：
                # 冻结主干时，1521 类 × 6 视图跑 220 万样本 top-1 只有 0.615，
                # 每类 1 视图同样预算能到 0.966（Analysis/pretrain_head_probe/joint_batch_structure.py）。
                # The last batch of a sampling round is short, so the view blocks
                # are strided by the actual batch size, not the configured one.
                taxon_batch_size = batch_taxon_labels.shape[0]
                taxon_start = (n_iter % n_views) * taxon_batch_size
                logits = self.model.classify(
                    embeddings[taxon_start:taxon_start + taxon_batch_size])
                taxon_labels = batch_taxon_labels
                loss_contrastive = genome_supervised_contrastive_loss(
                    projections, all_labels, self.contrastive_temperature)
                loss_taxon = self.criterion(logits, taxon_labels)
                loss_pair, pair_accuracy = None, None
                if self.pair_head is not None and self.pair_head_weight > 0:
                    loss_pair, pair_accuracy = pair_loss(
                        self.pair_head, embeddings, batch_taxon_labels,
                        self.pair_head_pairs, n_views,
                    )
                loss = (self.contrastive_loss_weight * loss_contrastive
                        + self.classification_loss_weight * loss_taxon)
                if loss_pair is not None:
                    loss = loss + self.pair_head_weight * loss_pair
                loss.backward()
                self.optimizer.step()
                running["loss"] += loss.item()
                running["supcon"] += loss_contrastive.item()
                running["classification"] += loss_taxon.item()
                if loss_pair is not None:
                    running["pair"] += loss_pair.item()
                    running["pair_accuracy"] += pair_accuracy.item()
                running["positive_cosine"] += (
                    projections[:len(batch_taxon_labels)]
                    * projections[len(batch_taxon_labels):2 * len(batch_taxon_labels)]
                ).sum(dim=-1).mean().item()
                for bin_index in torch.cat(bins).tolist():
                    length_bin_counts[int(bin_index)] += 1
                for pair_type in pair_types:
                    pair_type_counts[str(pair_type)] += 1
                if n_iter % self.log_every_n_steps == 0:
                    acc1_taxon, acc5_taxon = accuracy(logits, taxon_labels, topk=(1, 5))
                    supcon_acc1 = genome_contrastive_top1_accuracy(projections, all_labels)
                    supcon_acc1_sum += supcon_acc1
                    supcon_acc1_count += 1
                    message = (f"--> Epoch: {epoch_counter} / {self.epochs} "
                               f"LossSum:{loss.item():.4f}"
                               f"|LossSupCon:{loss_contrastive.item():.3f}"
                               f"|SupCon_Acc_1:{supcon_acc1 * 100:.2f}"
                               f"|LossTaxon:{loss_taxon.item():.3f}"
                               f"|Taxon_Acc_1:{acc1_taxon.item():.2f}"
                               f"|Taxon_Acc_5:{acc5_taxon.item():.2f}")
                    if loss_pair is not None:
                        message += (f"|LossPair:{loss_pair.item():.3f}"
                                    f"|Pair_Acc:{pair_accuracy.item() * 100:.2f}")
                    # 所有参数组共用同一条 warmup+cosine 曲线，但可以有各自的基准值
                    # （pretrain.py 会把分类头单独成一组、放在最后）。
                    group_lrs = [group["lr"] for group in self.optimizer.param_groups]
                    message += f"|LR: {group_lrs[0]:.8f}"
                    if len(group_lrs) > 1 and group_lrs[-1] != group_lrs[0]:
                        message += f"|HeadLR: {group_lrs[-1]:.3e}"
                    logger.info(message)
                n_iter += 1
            self.scheduler.step()
            self.model.eval()
            test_metrics = self.test_retrieval(test_loader)
            train_ref_metrics = self.train_reference_accuracy(train_reference_loader)
            self.model.train()
            # 选择 best checkpoint 的分数：分类现在（CE 权重 1.0）是主损失，所以
            # train-ref 占 0.7，留出基因组的短片段检索占 0.3。
            # 不再用 Pair AUPRC——它是在未归一化的点积上算的，波动大又容易被向量
            # 尺度的变化带偏。
            selection_score = (0.3 * test_metrics["short_macro_recall_at_1"]
                               + 0.7 * train_ref_metrics["top1_accuracy"])
            selection_key = (
                selection_score,
                test_metrics["short_macro_recall_at_1"],
                train_ref_metrics["top1_accuracy"],
                -test_metrics["retrieval_loss"],
            )
            train_summary = {key: value / max(n_iter, 1) for key, value in running.items()}
            train_summary["supcon_acc1"] = supcon_acc1_sum / max(supcon_acc1_count, 1)
            epoch_metrics = {
                "epoch": epoch_counter,
                "selection_score": selection_score,
                "test": test_metrics,
                "train_reference": train_ref_metrics,
                "train": train_summary,
                "sampling": {
                    "length_bin_counts": length_bin_counts,
                    "pair_type_counts": dict(pair_type_counts),
                },
            }
            loss_record[epoch_counter] = epoch_metrics
            is_best = best_key is None or selection_key > best_key
            if is_best:
                best_key = selection_key
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch_counter,
                "metrics": epoch_metrics,
                "num_classes": getattr(self.model, "classes_num", None),
                "metadata": self.checkpoint_metadata,
                "best_key": list(best_key),
            }
            if self.pair_head is not None:
                checkpoint["pair_head_state_dict"] = self.pair_head.state_dict()
            epoch_path = os.path.join(self.model_save_folder, f"checkpoint_epoch_{epoch_counter:03d}.pth")
            torch.save(checkpoint, epoch_path)
            torch.save(checkpoint, os.path.join(self.model_save_folder, "checkpoint_last.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(self.model_save_folder, "checkpoint_best.pth"))
            metrics_jsonl_path = os.path.join(self.model_save_folder, "test_metrics.jsonl")
            with open(metrics_jsonl_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
            # markdown 表每次按 jsonl 全量重渲染，保证所有行的列宽一致（对齐）
            render_pretrain_metrics_markdown(
                metrics_jsonl_path,
                os.path.join(self.model_save_folder, "test_metrics.md"),
            )
            logger.info(
                f"--> Epoch {epoch_counter} evaluation: score={selection_score:.6f}, "
                f"test_short_R1={test_metrics['short_macro_recall_at_1']:.6f}, "
                f"pair_auprc={test_metrics['pair_auprc']:.6f}, "
                f"train_ref_acc1={train_ref_metrics['top1_accuracy']:.6f}"
            )
        logger.info("--> Training has finished.")
        return loss_record

    @staticmethod
    def _binary_auc(scores, labels):
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        positives = int(labels.sum())
        negatives = int(labels.size - positives)
        if positives == 0 or negatives == 0:
            return 0.0, 0.0
        order = np.argsort(-scores, kind="mergesort")
        sorted_labels = labels[order]
        tp = np.cumsum(sorted_labels)
        fp = np.cumsum(1 - sorted_labels)
        tpr = np.concatenate([[0.0], tp / positives, [1.0]])
        fpr = np.concatenate([[0.0], fp / negatives, [1.0]])
        auroc = float(np.trapz(tpr, fpr))
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / positives
        auprc = float(np.sum((recall - np.concatenate([[0.0], recall[:-1]])) * precision))
        return auroc, auprc

    def test_retrieval(self, test_loader):
        self.model.eval()
        embedding_batches = []
        genome_batches = []
        length_bin_batches = []
        with torch.no_grad():
            for tokens, genome_ids, length_bins in tqdm(test_loader):
                tokens = tokens.to(torch.float32).to(self.device, non_blocking=True)
                embedding_batches.append(self.model.encode(tokens).cpu())
                genome_batches.append(genome_ids.long().cpu())
                length_bin_batches.append(length_bins.long().cpu())
        embeddings = torch.cat(embedding_batches, dim=0)
        genome_ids = torch.cat(genome_batches, dim=0)
        length_bins = torch.cat(length_bin_batches, dim=0)
        all_embeddings = embeddings.to(self.device)
        top1_correct = torch.zeros(len(embeddings), dtype=torch.bool)
        top5_correct = torch.zeros(len(embeddings), dtype=torch.bool)
        retrieval_loss_sum = 0.0
        chunk_size = 512
        with torch.no_grad():
            all_genomes_device = genome_ids.to(self.device)
            # encode 现在返回未归一化的向量，这里要比的是余弦相似度，
            # 所以先做 L2 归一化（原来 encode 自带归一化，这里就没写）。
            all_embeddings_unit = F.normalize(all_embeddings, dim=-1)
            for start in range(0, len(embeddings), chunk_size):
                end = min(start + chunk_size, len(embeddings))
                similarity = all_embeddings_unit[start:end] @ all_embeddings_unit.t()
                row = torch.arange(end - start, device=self.device)
                similarity[row, torch.arange(start, end, device=self.device)] = -float("inf")
                top_indices = similarity.topk(k=min(5, len(embeddings) - 1), dim=1).indices
                query_genomes = all_genomes_device[start:end]
                predicted_genomes = all_genomes_device[top_indices]
                top1_correct[start:end] = predicted_genomes[:, 0].eq(query_genomes).cpu()
                top5_correct[start:end] = predicted_genomes.eq(query_genomes[:, None]).any(dim=1).cpu()
                positive = all_genomes_device[None, :].eq(query_genomes[:, None])
                positive[row, torch.arange(start, end, device=self.device)] = False
                scaled_similarity = similarity / self.contrastive_temperature
                positive_logsumexp = torch.logsumexp(
                    scaled_similarity.masked_fill(~positive, -float("inf")), dim=1)
                all_logsumexp = torch.logsumexp(scaled_similarity, dim=1)
                retrieval_loss_sum += (all_logsumexp - positive_logsumexp).sum().item()

        def genome_macro_recall(query_mask):
            genome_scores = []
            for genome_id in genome_ids.unique():
                mask = genome_ids.eq(genome_id) & query_mask
                if mask.any():
                    genome_scores.append(top1_correct[mask].float().mean().item())
            return float(np.mean(genome_scores)) if genome_scores else 0.0

        bin_0_macro = genome_macro_recall(length_bins.eq(0))
        bin_1_macro = genome_macro_recall(length_bins.eq(1))
        short_macro = genome_macro_recall(length_bins < 2)

        # Deterministic sampled pair diagnostics avoid materializing every pair.
        # 打分一律用**归一化余弦**：原来这里用的是未归一化的点积，而训练过程中
        # 向量范数会大幅漂移（实测上游权重范数均值几百、我们 1–2），那会让
        # "同基因组对 vs 不同基因组对"的排序被尺度变化带偏，看起来像判别能力退化。
        rng = np.random.default_rng(2048)
        pair_scores = []
        pair_labels = []
        n_pairs = min(100_000, max(10_000, len(embeddings) * 10))
        emb_np = all_embeddings_unit.cpu().numpy()
        genome_np = genome_ids.numpy()
        for _ in range(n_pairs):
            left = int(rng.integers(0, len(embeddings)))
            right = int(rng.integers(0, len(embeddings) - 1))
            if right >= left:
                right += 1
            pair_scores.append(float(np.dot(emb_np[left], emb_np[right])))
            pair_labels.append(int(genome_np[left] == genome_np[right]))
        pair_auroc, pair_auprc = self._binary_auc(pair_scores, pair_labels)
        positive_scores = [s for s, label in zip(pair_scores, pair_labels) if label]
        negative_scores = [s for s, label in zip(pair_scores, pair_labels) if not label]

        metrics = {
            "short_macro_recall_at_1": short_macro,
            "length_768_1500_macro_recall_at_1": bin_0_macro,
            "length_1500_2500_macro_recall_at_1": bin_1_macro,
            "overall_recall_at_1": top1_correct.float().mean().item(),
            "overall_recall_at_5": top5_correct.float().mean().item(),
            "pair_auroc": pair_auroc,
            "pair_auprc": pair_auprc,
            "positive_cosine_mean": float(np.mean(positive_scores)) if positive_scores else 0.0,
            "negative_cosine_mean": float(np.mean(negative_scores)) if negative_scores else 0.0,
            "retrieval_loss": retrieval_loss_sum / len(embeddings),
            "num_contigs": len(embeddings),
            "num_genomes": int(genome_ids.unique().numel()),
        }
        return metrics

    def train_reference_accuracy(self, train_reference_loader):
        self.model.eval()
        total = 0
        total_loss = 0.0
        correct_1 = 0
        correct_5 = 0
        true_probability_sum = 0.0
        top1_probability_sum = 0.0
        seen_genomes = set()
        with torch.no_grad():
            for tokens, labels, _ in tqdm(train_reference_loader):
                tokens = tokens.to(torch.float32).to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()
                logits = self.model.classify(self.model.encode(tokens))
                total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
                probabilities = F.softmax(logits.float(), dim=-1)
                top_probabilities, top_indices = probabilities.topk(5, dim=-1)
                correct_1 += top_indices[:, 0].eq(labels).sum().item()
                correct_5 += top_indices.eq(labels[:, None]).any(dim=1).sum().item()
                true_probability_sum += probabilities.gather(1, labels[:, None]).sum().item()
                top1_probability_sum += top_probabilities[:, 0].sum().item()
                total += labels.numel()
                seen_genomes.update(labels.cpu().tolist())
        if total == 0:
            raise ValueError("empty training-reference evaluation loader")
        return {
            "classification_loss": total_loss / total,
            "top1_accuracy": correct_1 / total,
            "top5_accuracy": correct_5 / total,
            "mean_true_class_probability": true_probability_sum / total,
            "mean_top1_probability": top1_probability_sum / total,
            "num_contigs": total,
            "num_genomes": len(seen_genomes),
        }


class PretrainVQVAETrainer(object):

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 device: str,
                 epochs: int,
                 model_save_folder: str,
                 batch_size: int,
                 log_every_n_steps: int = 20
                 ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.model_save_folder = model_save_folder
        self.log_every_n_steps = log_every_n_steps
        self.batch_size = batch_size
        self.esp = 1e-9
        self.criteria = nn.KLDivLoss(reduction="batchmean")

    def custom_KL(self, y_pred, y_true):
        y_pred = torch.flatten(y_pred, end_dim=1)
        y_pred = F.log_softmax(y_pred, dim=1)
        y_true = torch.flatten(y_true, end_dim=1)
        # print(y_true, y_true.sum(dim=-1))
        return self.criteria(y_pred, y_true)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader,  model_weight_path=None):
        logger.info(f"--> Start VQ-VAE training with {self.epochs} epochs. Cross entropy loss. 1024.")
        logger.info(f"--> Training with {self.device} device.")
        loss_record = {}
        if model_weight_path is not None:
            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
            logger.info(f"--> Model weight has been loaded.")
        for epoch_counter in range(1, self.epochs + 1):
            n_iter = 0.
            self.model.train()
            for batch_genome_seq_tokens, _, _ in tqdm(train_loader):
                batch_genome_seq_tokens = batch_genome_seq_tokens.to(torch.float32).to(self.device, non_blocking=True)
                # ============ multi-res forward passes ... ============
                self.optimizer.zero_grad()
                x_hat, commit_loss, indices, logit = self.model(batch_genome_seq_tokens)
                # rec_loss = F.mse_loss(batch_genome_seq_tokens, x_hat, reduction="sum") / self.batch_size * 20.
                rec_loss = self.custom_KL(logit, batch_genome_seq_tokens) * 4.
                commit_loss *= 10.0
                loss = rec_loss + commit_loss
                loss.backward()
                self.optimizer.step()
                # logger
                if n_iter % self.log_every_n_steps == 0:
                    logger.info(f"--> Epoch: {epoch_counter} / {self.epochs} LossSum:{loss.item():.4f}" +
                                f"|Rec Loss: {rec_loss.item():.4f}" +
                                f"|Commit Loss: {commit_loss.item():.4f}" +
                                f"|LR: {self.optimizer.param_groups[0]['lr']:.8f}")
                n_iter += 1
            self.scheduler.step()
            self.model.eval()
            losses_valid = self.valid(valid_loader)
            self.model.train()
            loss_record[epoch_counter] = losses_valid
            # save model checkpoints
            checkpoint_name = f'checkpoint_{epoch_counter}_loss_{losses_valid:.4f}.pth'
            torch.save(self.model.state_dict(), os.path.join(self.model_save_folder, checkpoint_name))
            logger.info(f"--> Model checkpoint has been saved at {os.path.join(self.model_save_folder, checkpoint_name)}.")
        logger.info("--> Training has finished.")
        return loss_record

    def valid(self, valid_loader):
        losses = 0.
        self.model.eval()
        self.optimizer.zero_grad()
        with torch.no_grad():
            index = 0
            for batch_genome_seq_tokens, _, seq_len in tqdm(valid_loader):
                batch_genome_seq_tokens = batch_genome_seq_tokens.to(torch.float32).to(self.device, non_blocking=True)
                # ============ multi-res forward passes ... ============
                x_hat, commit_loss, indices, logit = self.model(batch_genome_seq_tokens)
                rec_loss = self.custom_KL(logit, batch_genome_seq_tokens) * 4.
                commit_loss *= 10.0
                loss = rec_loss + commit_loss
                losses += loss.item()
                index += 1
        logger.info(f"--> Valid loss is {losses / index + 0.}")
        return losses / index + 0.
