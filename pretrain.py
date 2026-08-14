import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from CompleteBin.logger import get_logger
from CompleteBin.Model.layers import CompleteBinBaseModel
from CompleteBin.Seqs.seq_utils import generate_feature_mapping_reverse
from CompleteBin.Trainer.dataset import PretrainDataset
from CompleteBin.Trainer.trainer import PretrainTrainer
from CompleteBin.Trainer.warmup import GradualWarmupScheduler
from CompleteBin.Trainer.optimizer import SingleDeviceMuonWithAuxAdam


logger = get_logger()


if __name__ == "__main__":
    count_kmer = 4
    dropout = 0.2
    lr = 1e-4
    lr_multiple = 8
    lr_warmup_epoch = 2
    train_epoch = 100
    weight_decay = 1e-3
    split_parts_list = [1, 11]
    min_seq_length = 768
    max_seq_length = 50000
    batch_size = 256
    device = "cuda"
    model_save_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CompleteBin-v1.1.1.0-Revision-last/CheckPoint"
    genome_db_folder_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Binner_DB/genomes-data"
    model_weight_load = None
    
    ## model trainer
    print(f"ssssddd,-600 {device}")
    classes_num = len(os.listdir(genome_db_folder_path)) + 1
    print(f"num classes: {classes_num}")
    count_kmer_dict_rev, count_nr_feature_rev = generate_feature_mapping_reverse(count_kmer)
    model = CompleteBinBaseModel(count_nr_feature_rev, classes_num, split_parts_list, dropout, False, hidden_dim=768, layers=3).to(device)
    if model_weight_load is not None:
        model.load_state_dict(torch.load(model_weight_load, map_location=device), strict=True)

    hidden_matrix_params = [p for p in model.parameters() if p.ndim >= 2]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    
    adam_groups = [dict(params=scalar_params, lr=lr)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=lr, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    
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
                                num_workers=40,
                                pin_memory=True,
                                drop_last=False)
    valid_set = PretrainDataset(genome_db_folder_path, count_kmer, split_parts_list, "valid", min_seq_length, max_seq_length)
    valid_loader = DataLoader(valid_set,
                            batch_size,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=False)
    trainer = PretrainTrainer(model, optimizer, warmUpScheduler, device, train_epoch,
                              model_save_folder, batch_size, log_every_n_steps=16)
    trainer.train(training_loader, valid_loader)