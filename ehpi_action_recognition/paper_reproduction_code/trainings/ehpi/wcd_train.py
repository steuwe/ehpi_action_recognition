import os
import random
import glob
import pandas as pd
import pdb
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
from ehpi_action_recognition.ehpi_action_recognition.paper_reproduction_code.trainings.ehpi.wcd_trainer import Trainer_Ehpi

def train(model_path: str, num_epochs: int, seed: int, split: int, end_idx, seq_length=32):
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
    root_dir = '/content/drive/My Drive/wcd_action_videos/action_frames_by_class/train/'
    test_dir = '/content/drive/My Drive/wcd_action_videos/action_frames_by_class/test/'
    class_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]

    class_image_paths = []
    end_idx = []
    for c, class_path in enumerate(class_paths):
        for d in os.scandir(class_path):
            if d.is_dir:
                paths = sorted(glob.glob(os.path.join(d.path, '*.jpg')))
                # Add class idx to paths
                paths = [(p, c) for p in paths]
                class_image_paths.extend(paths)
                end_idx.extend([len(paths)])

    end_idx = [0, *end_idx]
    end_idx = torch.cumsum(torch.tensor(end_idx), 0)
    seq_length = 32
    
    test_class_paths = [d.path for d in os.scandir(test_root_dir) if d.is_dir]

    test_class_image_paths = []
    test_end_idx = []
    for c, class_path in enumerate(test_class_paths):
        for d in os.scandir(test_class_path):
            if d.is_dir:
                test_paths = sorted(glob.glob(os.path.join(d.path, '*.jpg')))
                # Add class idx to paths
                test_paths = [(p, c) for p in test_paths]
                test_class_image_paths.extend(test_paths)
                test_end_idx.extend([len(test_paths)])

    test_end_idx = [0, *test_end_idx]
    test_end_idx = torch.cumsum(torch.tensor(test_end_idx), 0)
    
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
              num_epochs=50,
              seed=seed,
              split=1,
              end_idx, 
              seq_length)

    print("Train WCD")
    for split in range(1, 4):
        for seed in seeds:
            print("Split: {}, Seed: {}".format(split, seed))
            set_seed(seed)

            train(model_path=os.path.join(models_dir, "train_jhmdb"),
                  num_epochs=50,
                  seed=seed,
                  split=split,
                  end_idx,
                  seq_length)
