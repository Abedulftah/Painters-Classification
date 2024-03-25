from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from PIL import Image
import numpy as np


class ImagesDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path + '/train_info.csv')
        self.number_of_classes = len(self.unique_labels())
        self.labels = self.enumerate_labels()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        artist = self.data.iloc[idx]['artist']
        date = self.data.iloc[idx]['date']
        genre = self.data.iloc[idx]['genre']
        style = self.data.iloc[idx]['style']
        title = self.data.iloc[idx]['title']
        file_name = self.data.iloc[idx]['filename']
        return np.array(Image.open(self.file_path + f'/{file_name}').resize(())), self.labels[artist], artist, file_name

    def enumerate_labels(self):
        c = 0
        d = {}
        labels = self.unique_labels()
        for artist_name in labels:
            d[artist_name] = c
            c += 1
        return d

    def unique_labels(self):
        labels = self.data['artist'].dropna().tolist()
        return set(labels)

    def print_stats(self):
        print("#######################")
        print("dataset length: ", len(self.data))
        print("number of labels: ", len(self.unique_labels()))
        print("#######################")