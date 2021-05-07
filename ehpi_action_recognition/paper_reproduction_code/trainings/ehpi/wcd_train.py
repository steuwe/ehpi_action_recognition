import os
import random
from typing import List

import numpy as np
import torch
from nobos_commons.data_structures.dimension import ImageSize
from nobos_torch_lib.configs.training_configs.training_config_base import TrainingConfigBase
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import EhpiDataset, FlipEhpi, ScaleEhpi, \
    TranslateEhpi, NormalizeEhpi, RemoveJointsOutsideImgEhpi, RemoveJointsEhpi
from wcd_dataset import MyDataset, MySampler
from nobos_torch_lib.learning_rate_schedulers.learning_rate_scheduler_stepwise import \
    LearningRateSchedulerStepwise
from nobos_torch_lib.models.action_recognition_models.ehpi_small_net import EHPISmallNet
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ehpi_action_recognition.config import ehpi_dataset_path, models_dir
from ehpi_action_recognition.trainer_ehpi import TrainerEhpi

def train(model_path: str, num_epochs: int, seed: int, split: int):
    # Train set
    sampler = MySampler(end_idx, seq_length)
    transform = transforms.Compose([
        transforms.Resize((1280, 720)),
        transforms.ToTensor()
    ])

    dataset = MyDataset(
      image_paths=class_image_paths,
      seq_length=seq_length,
      transform=transform,
      length=len(sampler))

    train_loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler
    )

    # config
    train_config = TrainingConfigBase("ehpi_jhmdb_{}_split_{}".format(seed, split), model_path)
    train_config.learning_rate = lr
    train_config.learning_rate_scheduler = LearningRateSchedulerStepwise(lr_decay=0.1, lr_decay_epoch=50)
    train_config.weight_decay = weight_decay
    train_config.num_epochs = num_epochs
    train_config.checkpoint_epoch = num_epochs

    trainer = TrainerEhpi()
    trainer.train(train_loader, train_config, model=EHPISmallNet(21))

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(0)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    image_size = ImageSize(1280, 720)
    seeds = [0, 104, 123, 142, 200]
    batch_size = 64
    weight_decay = 5e-4
    lr = 0.05

    print("Train WCD")
    for seed in seeds:
        print("Seed: {}".format(seed))
        set_seed(seed)

        train(model_path=os.path.join(models_dir, "train_jhmdb_gt"),
              num_epochs=140,
              seed=seed,
              split=1)

    print("Train WCD")
    for split in range(1, 4):
        for seed in seeds:
            print("Split: {}, Seed: {}".format(split, seed))
            set_seed(seed)

            train(model_path=os.path.join(models_dir, "train_jhmdb"),
                  num_epochs=200,
                  seed=seed,
                  split=split)
