"""Second-stage refit of the reference-genome classifier head.

The contrastive objective owns the backbone and trains it well: with a frozen
encoder, a nearest-prototype classifier over the same embeddings already
reaches ~0.70 top-1 over 13,773 reference genomes. The head that is trained
jointly with that objective does not get there - it shares the backbone's tiny
learning rate, and raising that rate inside the contrastive loop is unstable,
because every step pushes all ~1.4e4 class rows (Adam normalises those tiny
pushes to full size) while only the ~64 classes present in the batch receive a
signal.

So the head is fitted afterwards, on cached embeddings, with a normal learning
rate and a batch that covers a large share of the classes at once:

    refit lr 3e-3, 6 epochs, 8 fragments per genome
    top-1: 0.617 (jointly trained head) -> 0.923
           0.341 -> 0.947 for 768-1500 bp fragments

（上面那组数字还是余弦头时代测的。现在分类头是**裸线性层** logits = W·x，
输入就是主干输出的原始向量，没有 LayerNorm、没有余弦温度。
本模块的重拟合就是在这份原始特征上做普通线性探针，和部署时的 ``classify()`` 一致。）

The backbone is left untouched (parameters frozen), so retrieval quality and
the downstream binning behaviour are unchanged; only the confidence signal and
the same-genome mask that consumes it improve.
"""

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from CompleteBin.DataProcess.data_utils import get_features_of_one_seq
from CompleteBin.Seqs.seq_utils import generate_feature_mapping_reverse
from CompleteBin.Trainer.dataset import PretrainDataset
from CompleteBin.logger import get_logger

logger = get_logger()

DEFAULT_SPLIT_PARTS = [1, 11]
DEFAULT_COUNT_KMER = 4

_KMER_CACHE = {}


def _tokens(sequence, count_kmer, split_parts_list, min_len):
    cached = _KMER_CACHE.get(count_kmer)
    if cached is None:
        cached = generate_feature_mapping_reverse(count_kmer)
        _KMER_CACHE[count_kmer] = cached
    kmer_dict, nr_features = cached
    # 4-mer 组成向量，136 维
    tokens, _, _, _ = get_features_of_one_seq(
        sequence, None, count_kmer, kmer_dict, nr_features, split_parts_list,
        min_len,
    )
    return tokens.astype(np.float32, copy=False)


def _sample_fragments(job):
    """Read one genome once and return many gallery fragments plus one query."""
    (record, gallery_seeds, query_seed, count_kmer, split_parts_list,
     min_len, max_len, dtype) = job
    dataset = PretrainDataset([record], count_kmer, split_parts_list, "train_eval",
                              min_len, max_len, fixed_seed=query_seed)
    sequences = dataset._read_fasta_sequences(record["path"])
    if not sequences:
        raise ValueError(f"no eligible sequence in {record['path']}")
    gallery = []
    for seed in gallery_seeds:
        sample, _, _ = dataset._sample_contig_any_length(sequences, random.Random(seed))
        gallery.append(_tokens(sample, count_kmer, split_parts_list, min_len))
    query, _, _ = dataset._sample_contig_any_length(sequences, random.Random(query_seed))
    gallery = np.stack(gallery, axis=0).astype(dtype, copy=False)
    return gallery, _tokens(query, count_kmer, split_parts_list, min_len), len(query)


def build_gallery(records, device, *, gallery_per_genome=8, workers=64, count_kmer=DEFAULT_COUNT_KMER,
                  split_parts_list=None, min_len=768, max_len=50000, seed=2048, chunk_size=256,
                  dtype=np.float16, progress_every=None):
    """Sample and tokenise the refit gallery and the evaluation queries."""
    split_parts_list = split_parts_list or DEFAULT_SPLIT_PARTS
    jobs = [
        (record,
         # The evaluation query uses ``seed + 104729 * index`` (the same query as
         # train_reference_accuracy), so the gallery must avoid that seed: with
         # view 0 sharing it, every genome's query was inside its own gallery and
         # the measured accuracy was inflated (a 1-fragment gallery scored 1.0).
         [seed + 104729 * index + 7919 * (view + 1) for view in range(gallery_per_genome)],
         seed + 104729 * index,
         count_kmer, split_parts_list, min_len, max_len, dtype)
        for index, record in enumerate(records)
    ]
    total = len(jobs)
    # token 宽度由 k-mer 长度决定（4-mer → 136，5-mer → 512），不能写死。
    # 每条片段固定 6 个 token：整条片段 + get_features_of_one_seq 里的 5 个子片段。
    token_dim = generate_feature_mapping_reverse(count_kmer)[1]
    # Pre-allocate: with the 10x sampling used in pretraining the gallery is tens
    # of gigabytes, and concatenating per-chunk blocks would double that spike.
    gallery_out = np.empty((total, gallery_per_genome, 6, token_dim), dtype=dtype)
    query_out = np.empty((total, 6, token_dim), dtype=np.float32)
    lengths_out = np.empty(total, dtype=np.int64)
    chunk_size = max(4, min(chunk_size, max(4, 8192 // max(gallery_per_genome, 1))))
    progress_every = progress_every or chunk_size * 4
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(0, len(jobs), chunk_size):
            chunk = jobs[start:start + chunk_size]
            results = (executor.map(_sample_fragments, chunk, chunksize=4) if executor
                       else map(_sample_fragments, chunk))
            results = list(results)
            stop = start + len(results)
            for offset, (gallery, query, length) in enumerate(results):
                gallery_out[start + offset] = gallery
                query_out[start + offset] = query
                lengths_out[start + offset] = length
            sampled = min(start + chunk_size, len(jobs))
            if sampled % progress_every < chunk_size or sampled == len(jobs):
                logger.info(f"--> Head refit gallery: {sampled}/{len(jobs)} genomes sampled "
                            f"({gallery_per_genome} fragments each).")
    finally:
        if executor is not None:
            executor.shutdown()
    return gallery_out, query_out, lengths_out


@torch.no_grad()
def embed(model, token_blocks, device, batch_size=4096):
    """Embed a [N, ...] block of token arrays with the frozen backbone."""
    flat = token_blocks.reshape(-1, *token_blocks.shape[-2:])
    vectors = np.empty((len(flat), model.pos_embedding.shape[-1]), dtype=np.float32)
    for start in range(0, len(flat), batch_size):
        batch = torch.as_tensor(np.stack(flat[start:start + batch_size]),
                                dtype=torch.float32, device=device)
        vectors[start:start + len(batch)] = _backbone_features(model, batch).cpu().numpy()
    return vectors


def _backbone_features(model, tokens):
    """backbone 输出 → 分类头真正看到的特征。

    分类头现在是裸线性层，所以这里就是主干原始向量；如果哪天又给分类头加了
    前置层（比如 LayerNorm），在这里补上，保证所有"用分类头算 logits"的地方
    和部署时是同一套特征。
    """
    features = model.encode(tokens)
    if getattr(model, "out_norm", None) is not None:
        features = model.out_norm(features)
    return features


@torch.no_grad()
def _evaluate(weight, features, labels):
    """线性探针打分：features 就是分类头看到的特征（现在是主干原始向量）。

    直接和权重做点积，不做权重的 L2 归一化、不除温度——和部署时 ``classify()``
    （即 ``out_linear(x)``）完全一致。
    """
    logits = features.float() @ weight.t()
    probabilities = F.softmax(logits.float(), dim=-1)
    # top-5 只在类别数够时成立；小规模冒烟测试（类别 < 5）也能跑。
    k = min(5, probabilities.shape[-1])
    values, indices = probabilities.topk(k, dim=-1)
    top1 = values[:, 0]
    return {
        "top1_accuracy": float(indices[:, 0].eq(labels).float().mean().item()),
        "top5_accuracy": float(indices.eq(labels[:, None]).any(dim=1).float().mean().item()),
        "mean_top1_probability": float(top1.mean().item()),
        "top1_probability": top1.cpu().numpy(),
    }


def refit_classifier_head(model, records, device, *, gallery_per_genome=8, epochs=6, lr=3e-3,
                          batch_size=4096, workers=64, seed=2048,
                          count_kmer=DEFAULT_COUNT_KMER, split_parts_list=None,
                          min_len=768, max_len=50000, save_path=None):
    refitter = OnlineHeadRefitter(
        model, records, device, fragments_per_genome=gallery_per_genome, fit_epochs=epochs,
        lr=lr, batch_size=batch_size, workers=workers, seed=seed,
        count_kmer=count_kmer, split_parts_list=split_parts_list, min_len=min_len,
        max_len=max_len,
    )
    refitter.build()
    metrics = refitter.refit()
    if save_path:
        torch.save(model.state_dict(), save_path)
    return metrics


class OnlineHeadRefitter(object):
    """Alternating schedule: refit the classifier head on frozen-backbone embeddings.

    The contrastive loop cannot train this head (see the module docstring and
    Analysis/pretrain_head_probe/FINDINGS.md), so the head is fitted separately:
    the backbone is frozen, every genome contributes ``fragments_per_genome``
    cached fragments, and the head is trained at a normal learning rate. The
    gallery tokens are sampled once and reused, so each later refit only costs a
    re-embedding pass plus the head fit.
    """

    def __init__(self, model, records, device, *, fragments_per_genome=400, fit_epochs=6,
                 lr=3e-3, batch_size=4096, embed_batch_size=8192,
                 workers=64, seed=2048, count_kmer=DEFAULT_COUNT_KMER, split_parts_list=None,
                 min_len=768, max_len=50000, cache_dir=None, warm_start=True,
                 lr_schedule="cosine", total_epochs=None):
        self.model = model
        self.records = list(records)
        self.device = device
        self.fragments_per_genome = int(fragments_per_genome)
        self.fit_epochs = int(fit_epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.embed_batch_size = int(embed_batch_size)
        self.workers = int(workers)
        self.seed = int(seed)
        self.count_kmer = count_kmer
        self.split_parts_list = split_parts_list or DEFAULT_SPLIT_PARTS
        self.min_len = min_len
        self.max_len = max_len
        # Sampling the gallery costs ~15 minutes for 13,773 genomes x 400
        # fragments, so it is cached on disk and re-used across restarts.
        self.cache_dir = cache_dir
        # Warm start: every refit continues from the head as it stands now (which
        # is the previous refit's result) instead of re-deriving prototypes, and
        # the Adam moments are kept across refits. lr_schedule then decays the
        # learning rate over the whole run, so late refits only refine.
        self.warm_start = bool(warm_start)
        self.lr_schedule = lr_schedule
        self.total_epochs = int(total_epochs) if total_epochs else None
        self.weight = None
        self.optimizer = None
        self.last_lr = None
        self.gallery_tokens = None
        self.query_tokens = None
        self.query_lengths = None
        self.history = []

    def build(self):
        """Sample and tokenise the refit gallery once."""
        if self.cache_dir and os.path.exists(self._cache_path("gallery")):
            logger.info(f"--> Head refit gallery: loading cache from {self.cache_dir}.")
            self.gallery_tokens = np.load(self._cache_path("gallery"))
            self.query_tokens = np.load(self._cache_path("queries"))
            self.query_lengths = np.load(self._cache_path("lengths"))
            logger.info(f"--> Head refit gallery ready (cached): {self.gallery_tokens.shape}.")
            return
        logger.info(
            f"--> Head refit gallery: {len(self.records)} genomes x "
            f"{self.fragments_per_genome} fragments."
        )
        gallery, queries, lengths = build_gallery(
            self.records, self.device, gallery_per_genome=self.fragments_per_genome,
            workers=self.workers, count_kmer=self.count_kmer,
            split_parts_list=self.split_parts_list, min_len=self.min_len,
            max_len=self.max_len, seed=self.seed,
        )
        self.gallery_tokens = gallery
        self.query_tokens = queries
        self.query_lengths = lengths
        logger.info(f"--> Head refit gallery ready: {gallery.shape[0]} genomes, "
                    f"tokens {gallery.shape}.")
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(self._cache_path("gallery"), gallery)
            np.save(self._cache_path("queries"), queries)
            np.save(self._cache_path("lengths"), lengths)
            logger.info(f"--> Head refit gallery cached at {self.cache_dir}.")

    def _cache_path(self, name):
        # Version bumped when the sampling seeds changed (gallery must not
        # overlap the evaluation query).
        key = (f"v2_{len(self.records)}g_{self.fragments_per_genome}f_{self.seed}s_"
               f"{self.min_len}_{self.max_len}")
        return os.path.join(self.cache_dir, f"{name}_{key}.npy")

    @torch.no_grad()
    def _embed(self, tokens, batch_size=None):
        """Embed an [N, ...] token block with the (frozen) backbone."""
        batch_size = batch_size or self.embed_batch_size
        flat = tokens.reshape(-1, *tokens.shape[-2:])
        out = np.empty((len(flat), self.model.pos_embedding.shape[-1]), dtype=np.float16)
        for start in range(0, len(flat), batch_size):
            batch = torch.as_tensor(np.asarray(flat[start:start + batch_size], dtype=np.float32),
                                    device=self.device)
            with torch.no_grad():
                feats = _backbone_features(self.model, batch)
            out[start:start + len(batch)] = feats.cpu().numpy().astype(np.float16)
        return out

    def current_lr(self, epoch=None, total_epochs=None):
        """Learning rate of the next refit, given the run-wide schedule."""
        total = total_epochs or self.total_epochs
        if self.lr_schedule != "cosine" or not epoch or not total or total <= 1:
            return self.lr
        # (epoch - 1) / total rather than / (total - 1) so the final epoch keeps a
        # small non-zero lr instead of freezing completely.
        progress = min(max((epoch - 1) / float(total), 0.0), 1.0)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _fit_head(self, gallery_embeddings, labels, queries, epoch=None, total_epochs=None):
        num_classes = len(self.records)
        # Keep the gallery on the host: with the 10x sampling asked for in
        # pretraining it is far larger than the GPU should hold.
        host_gallery = torch.as_tensor(
            gallery_embeddings.reshape(num_classes, self.fragments_per_genome, -1)
            .astype(np.float32)
        )
        # 线性探针：类别均值直接当权重初值（最近类均值），不做 L2 归一化——
        # 部署时的分类头就是裸线性层的普通点积。
        prototypes = host_gallery.mean(dim=1).to(self.device)
        prototype_accuracy = _evaluate(prototypes, queries, labels)["top1_accuracy"]
        current_head = self.model.out_linear.weight.detach()
        if self.warm_start:
            # Continue from the head as it stands (the previous refit's result,
            # possibly nudged by the joint phase) rather than rebuilding it.
            starting_point = current_head.clone()
        else:
            starting_point = prototypes.clone()
        start_accuracy = _evaluate(starting_point, queries, labels)["top1_accuracy"]
        lr = self.current_lr(epoch, total_epochs)
        self.last_lr = lr
        if self.weight is None:
            self.weight = torch.nn.Parameter(starting_point.clone())
            self.optimizer = torch.optim.Adam([self.weight], lr=lr)
        else:
            with torch.no_grad():
                self.weight.copy_(starting_point)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        weight, optimizer = self.weight, self.optimizer
        logger.info(
            f"--> Head refit: {'warm' if self.warm_start else 'prototype'} start "
            f"query top-1={start_accuracy:.4f} (prototype k-NN={prototype_accuracy:.4f}), "
            f"lr={lr:.2e} on a {self.lr_schedule} schedule over "
            f"{total_epochs or self.total_epochs or 1} run epochs, "
            f"{self.fit_epochs} fit epochs."
        )
        flat_labels = labels.repeat_interleave(self.fragments_per_genome)
        total = len(flat_labels)
        generator = torch.Generator().manual_seed(self.seed)

        epoch_accuracy = []
        epoch_loss = []
        for fit_epoch in range(1, self.fit_epochs + 1):
            total_loss, steps = 0.0, 0
            order = torch.randperm(total, generator=generator)
            for start in range(0, total, self.batch_size):
                index = order[start:start + self.batch_size]
                # host_gallery 常驻内存，按行取出来后再搬到显存，避免 15000×8×768
                # 的样本池整个占住 GPU。
                batch = host_gallery.reshape(total, -1)[index].to(self.device)
                target = flat_labels[index.to(self.device)]
                logits = batch @ weight.t()
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            accuracy = _evaluate(weight, queries, labels)["top1_accuracy"]
            epoch_accuracy.append(accuracy)
            epoch_loss.append(total_loss / max(steps, 1))
            logger.info(f"--> Head refit epoch {fit_epoch}/{self.fit_epochs}: "
                        f"train CE={total_loss / max(steps, 1):.4f}, query top-1={accuracy:.4f}")
        return (weight.detach(), epoch_accuracy, epoch_loss, prototype_accuracy,
                start_accuracy, lr)

    def refit(self, epoch=None, total_epochs=None):
        """Freeze the backbone, refit the head, install it and report metrics."""
        if self.gallery_tokens is None:
            raise RuntimeError("call build() before refit()")
        was_training = self.model.training
        self.model.eval()
        labels = torch.arange(len(self.records), device=self.device)
        gallery_embeddings = self._embed(self.gallery_tokens)
        query_embeddings = torch.as_tensor(
            np.asarray(self._embed(self.query_tokens[:, None]), dtype=np.float32),
            device=self.device,
        )
        before = _evaluate(self.model.out_linear.weight.detach(),
                           query_embeddings, labels)
        (weight, epoch_accuracy, epoch_loss, prototype_accuracy,
         start_accuracy, lr) = self._fit_head(gallery_embeddings, labels, query_embeddings,
                                              epoch=epoch, total_epochs=total_epochs)
        after = _evaluate(weight, query_embeddings, labels)
        with torch.no_grad():
            self.model.out_linear.weight.copy_(weight)

        bins = [(768, 1500), (1500, 2500), (2500, 5000), (5000, 10000), (10000, self.max_len)]
        per_bin = []
        for low, high in bins:
            mask = (self.query_lengths >= low) & (self.query_lengths < high)
            if not mask.any():
                continue
            mask_tensor = torch.as_tensor(mask, device=self.device)
            per_bin.append({
                "range": f"{low}-{high}",
                "count": int(mask.sum()),
                "top1_accuracy": _evaluate(weight, query_embeddings[mask_tensor],
                                           labels[mask_tensor])["top1_accuracy"],
            })
        if was_training:
            self.model.train()
        metrics = {
            "fragments_per_genome": self.fragments_per_genome,
            "fit_epochs": self.fit_epochs,
            "lr": self.lr,
            "head_before": {key: value for key, value in before.items()
                            if key != "top1_probability"},
            "head_after": {key: value for key, value in after.items()
                           if key != "top1_probability"},
            "fit_epoch_top1": epoch_accuracy,
            "fit_epoch_ce": epoch_loss,
            "fit_final_ce": epoch_loss[-1] if epoch_loss else float("nan"),
            "prototype_knn_top1": prototype_accuracy,
            "start_top1": start_accuracy,
            "run_epoch": epoch,
            "lr": lr,
            "warm_start": self.warm_start,
            "per_length_bin": per_bin,
        }
        self.history.append(metrics)
        logger.info(
            f"--> Head refit done: top-1 {before['top1_accuracy']:.4f} -> "
            f"{after['top1_accuracy']:.4f}, mean top-1 probability "
            f"{before['mean_top1_probability']:.4f} -> {after['mean_top1_probability']:.4f}."
        )
        return metrics
