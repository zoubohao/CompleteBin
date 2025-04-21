import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from Src.logger import get_logger
from Src.Model.model import DeeperBinBaseModel
from Src.Seqs.seq_utils import generate_feature_mapping_reverse
from Src.Trainer.dataset import PretrainDataset
from Src.Trainer.trainer import PretrainTrainer
from Src.Trainer.warmup import GradualWarmupScheduler

logger = get_logger()


if __name__ == "__main__":
    count_kmer = 4
    dropout = 0.1
    lr = 1e-7
    lr_multiple = 5
    lr_warmup_epoch = 1
    train_epoch = 24
    weight_decay = 1e-4
    split_parts_list = [1, 16]
    min_seq_length = 768
    max_seq_length = 60000
    batch_size = 256
    device = "cuda:0"
    model_save_folder = "./DeepMetaBin-DB/CheckPoint/"
    genome_db_folder_path = "../Binner_DB/genomes-data/"
    model_weight_load = "./DeepMetaBin-DB/CheckPoint/pretrain_weight_hidden_dim_512_layers_3.pth"
    
    ## model trainer
    classes_num = len(os.listdir(genome_db_folder_path)) + 1
    count_kmer_dict_rev, count_nr_feature_rev = generate_feature_mapping_reverse(count_kmer)
    model = DeeperBinBaseModel(count_nr_feature_rev, classes_num, split_parts_list, dropout, hidden_dim=512, layers=3).to(device)
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
                                num_workers=16,
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
                              model_save_folder, batch_size,log_every_n_steps=20)
    trainer.train(training_loader, valid_loader)