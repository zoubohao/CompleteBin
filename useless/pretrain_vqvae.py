import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from CompleteBin.logger import get_logger
from CompleteBin.Seqs.seq_utils import generate_feature_mapping_reverse
from CompleteBin.Model.model import SimVQVAE
from CompleteBin.Trainer.dataset import PretrainDataset
from CompleteBin.Trainer.trainer import PretrainVQVAETrainer
from CompleteBin.Trainer.warmup import GradualWarmupScheduler


logger = get_logger()

# x = torch.randn(1, 1024, 512)
# quantized, indices, commit_loss = sim_vq(x)

# assert x.shape == quantized.shape
# assert torch.allclose(quantized, sim_vq.indices_to_codes(indices), atol = 1e-6)






if __name__ == "__main__":
    count_kmer = 4
    lr = 1e-7
    lr_multiple = 10
    lr_warmup_epoch = 2
    train_epoch = 24
    weight_decay = 1e-4
    split_parts_list = [1, 16]
    min_seq_length = 768
    max_seq_length = 100000
    batch_size = 128
    device = "cuda:1"
    model_save_folder = "./vqvae_ckpt/"
    genome_db_folder_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Binner_DB/genomes-data"
    model_weight_load = "/home/comp/21481598/CompleteBin-v1.1.0.0/vqvae_ckpt/checkpoint_64_loss_1.62.pth"
    
    ## model trainer
    classes_num = len(os.listdir(genome_db_folder_path)) + 1
    count_kmer_dict_rev, count_nr_feature_rev = generate_feature_mapping_reverse(count_kmer)
    model = SimVQVAE(136, split_parts_list, infer_mode=False)
    if model_weight_load is not None:
        model.load_state_dict(torch.load(model_weight_load, map_location=device), strict=True)
    parameters = model.parameters()
    optimizer = AdamW(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-6
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer,
        lr_multiple,
        lr_warmup_epoch,
        train_epoch - lr_warmup_epoch,
    )
    training_set = PretrainDataset(genome_db_folder_path, count_kmer, split_parts_list, "train", min_seq_length, max_seq_length)
    training_loader = DataLoader(training_set,
                                batch_size,
                                shuffle=True,
                                num_workers=128,
                                pin_memory=True,
                                drop_last=False)
    valid_set = PretrainDataset(genome_db_folder_path, count_kmer, split_parts_list, "valid", min_seq_length, max_seq_length)
    valid_loader = DataLoader(valid_set,
                            batch_size,
                            shuffle=False,
                            num_workers=32,
                            pin_memory=True,
                            drop_last=False)
    trainer = PretrainVQVAETrainer(model, optimizer, warmUpScheduler, device, train_epoch,
                              model_save_folder, batch_size,log_every_n_steps=10)
    trainer.train(training_loader, valid_loader)