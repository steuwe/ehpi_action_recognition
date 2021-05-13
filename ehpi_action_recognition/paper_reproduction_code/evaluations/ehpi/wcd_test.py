import os

from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import EhpiDataset, NormalizeEhpi, \
    RemoveJointsOutsideImgEhpi
from nobos_torch_lib.models.action_recognition_models.ehpi_small_net import EHPISmallNet
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ehpi_action_recognition.config import ehpi_dataset_path, models_dir
from ehpi_action_recognition.tester_ehpi import TesterEhpi

def test_gt(test_loader):
    seeds = [0, 104, 123, 142, 200]
    for seed in seeds:
        print("Test WCD GT on seed: {}".format(seed))
        weights_path = os.path.join(models_dir, "wcd_weights", "ehpi_wcd_{}_split_1_cp0140.pth".format(seed))

        # Test set
        tester = TesterEhpi()
        tester.test(test_loader, weights_path, model=EHPISmallNet(21))


def test_jhmdb():
    seeds = [0, 104, 123, 142, 200]
    for split in range(1, 4):
        for seed in seeds:
            print("Test JHMDB Split {} on seed: {}".format(split, seed))
            weights_path = os.path.join(models_dir, "jhmdb", "ehpi_jhmdb_{}_split_{}_cp0200.pth".format(seed, split))

            # Test set
            test_set = get_test_set(os.path.join(ehpi_dataset_path, "jhmdb", "JHMDB_ITSC-1-POSE/"), ImageSize(320, 240))
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

            tester = TesterEhpi()
            tester.test(test_loader, weights_path, model=EHPISmallNet(21))


if __name__ == '__main__':
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
    seq_length = 32
    
    test_sampler = MySampler(test_end_idx, seq_length)
    transform = transforms.Compose([
        transforms.Resize((1280, 720)),
        transforms.ToTensor()
    ])

    test_dataset = MyDataset(
      image_paths=test_class_image_paths,
      seq_length=seq_length,
      transform=transform,
      length=len(test_sampler))

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        sampler=sampler
    )
    
    test_gt(test_loader)
    test_jhmdb()
