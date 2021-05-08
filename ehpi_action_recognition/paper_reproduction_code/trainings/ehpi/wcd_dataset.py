import os
import glob
import pdb
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image


class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):        
        indices = []
        for i in range(len(end_idx)-1):
            start = end_idx[i]
            end = end_idx[i+1] - seq_length
            if end - start >= 32:
                indices.append(torch.arange(start, end))
        indices = torch.cat(indices)
        self.indices = indices
        
    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.indices)


class MyDataset(Dataset):
    def __init__(self, image_paths, seq_length, transform, length):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        
    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        images = []
        annots = pd.read_csv('/content/drive/My Drive/wcd_action_videos/annots_per_frame.csv')
        actions = []
        keypoints = []
        for i in indices:
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            actions.append(annots.loc[annots['file_name'] == image_path.split('/')[-1]]['action'].item())
            kpts = annots.loc[annots['file_name'] == image_path.split('/')[-1]]['keypoints'].item()
            kpts = kpts.replace('[', '').replace(']', '').split(' ')
            keypoints.append([float(x) for x in kpts if x != ''])
            if self.transform:
                image = self.transform(image)
            images.append(image)
        x = torch.stack(images)
        #y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        y = torch.tensor(actions[0])
        z = np.array(keypoints)
        z = np.reshape(z, (1, 32, 20, 3))
        z[:,:,:,2] = 0
        z = np.transpose(z, (0, 3, 1, 2))
        z = torch.tensor(z)
        return x, y, z
    
    def __len__(self):
        return self.length


root_dir = '/content/drive/My Drive/wcd_action_videos/action_frames_by_class/'
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
seq_length = 10

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

loader = DataLoader(
    dataset,
    batch_size=1,
    sampler=sampler
)
