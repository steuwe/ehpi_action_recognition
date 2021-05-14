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
        annots = pd.read_csv('/content/drive/My Drive/wcd_action_videos/annots_per_frame.csv')
        actions = []
        keypoints = []
        for i in indices:
            image_path = self.image_paths[i][0]
            actions.append(annots.loc[annots['file_name'] == image_path.split('/')[-1]]['action'].item())
            kpts = annots.loc[annots['file_name'] == image_path.split('/')[-1]]['keypoints'].item()
            kpts = kpts.replace('[', '').replace(']', '').split(' ')
            keypoints.append([float(x) for x in kpts if x != ''])
        y = torch.tensor(actions[0])
        x = np.array(keypoints)
        x = np.reshape(x, (32, 20, 3))
        x[:,:,2] = 0
        x[:,:,0] = x[:,:,0]/1280
        x[:,:,1] = x[:,:,1]/720
        x = np.transpose(x, (2, 0, 1))
        x = torch.tensor(x)
        return x, y
    
    def __len__(self):
        return self.length
