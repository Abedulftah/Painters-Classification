import os
import torch
from ImagesDataset import *
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
directory = r'train/train'

d = ImagesDataset(file_path=directory)
loader_train = DataLoader(d, batch_size=2, shuffle=True, pin_memory=True)