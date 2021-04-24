import os
import random
from typing import List

import torch
from ehpi_action_recognition.config import ehpi_dataset_path
from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_torch_lib.configs.training_configs.training_config_base import TrainingConfigBase
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import EhpiDataset, RemoveJointsOutsideImgEhpi, \
    ScaleEhpi, TranslateEhpi, FlipEhpi, NormalizeEhpi
from nobos_torch_lib.datasets.samplers.imbalanced_dataset_sampler import ImbalancedDatasetSampler
from nobos_torch_lib.models.detection_models.shufflenet_v2 import ShuffleNetV2
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import transforms

from ehpi_action_recognition.trainer_ehpi import TrainerEhpi

foot_indexes: List[int] = [17, 18, 19, 20]
knee_indexes: List[int] = [13, 14, 15, 16]


def get_train_set(dataset_path: str, image_size: ImageSize):
    num_joints = 20
    left_indexes: List[int] = [0, 5, 8, 10, 12, 14, 16, 18]
    right_indexes: List[int] = [1, 6, 9, 11, 13, 15, 17, 19]

    datasets: List[EhpiDataset] = [
        # Set 1
        EhpiDataset(os.path.join(dataset_path, "test"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints, dataset_part=DatasetPart.TEST),
        # Set 2
        EhpiDataset(os.path.join(dataset_path, "train"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints, dataset_part=DatasetPart.TRAIN),
    ]
    for dataset in datasets:
        dataset.print_label_statistics()

    return ConcatDataset(datasets)


if __name__ == '__main__':
    batch_size = 128
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Train set
    train_set = get_train_set(ehpi_dataset_path, image_size=ImageSize(1280, 720))
    sampler = ImbalancedDatasetSampler(train_set, dataset_type=EhpiDataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=1)

    # config
    train_config = TrainingConfigBase("ehpi_model", "models")
    train_config.weight_decay = 0
    train_config.num_epochs = 140

    trainer = TrainerEhpi()

    trainer.train(train_loader, train_config, model=ShuffleNetV2(3))
