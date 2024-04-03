import random

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import torch
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None


class ImagesDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path + r'\train\train\train_info.csv')
        self.small_data_set()
        self.labels = self.enumerate_labels()
        self.number_of_classes = len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        artist1 = self.data.iloc[idx]['artist']
        # date1 = self.data.iloc[idx]['date']
        # genre1 = self.data.iloc[idx]['genre']
        # style1 = self.data.iloc[idx]['style']
        # title1 = self.data.iloc[idx]['title']
        file_name1 = self.data.iloc[idx]['filename']
        image1 = Image.open(self.file_path + f'\\preprocessed_train\\{file_name1}')
        image1 = np.transpose(np.array(image1) / 255.0)
        label1 = self.labels[artist1]
        if random.random() > 0.5:
            # Get information about an image with a different label
            different_artist = random.choice(list(self.labels.keys()))
            while different_artist == artist1:
                different_artist = random.choice(list(self.labels.keys()))
            file_name2 = random.choice(self.data[self.data['artist'] == different_artist]['filename'].values)
        else:
            # Get information about another image with the same label
            file_name2 = file_name1
            while file_name2 == file_name1:
                file_name2 = random.choice(self.data[self.data['artist'] == artist1]['filename'].values)
        image2 = Image.open(self.file_path + f'\\preprocessed_train\\{file_name2}')
        image2 = np.transpose(np.array(image2) / 255.0)
        label2 = self.labels[self.data[self.data['filename'] == file_name2]['artist'].values[0]]
        similarity_label = 0 if label1 == label2 else 1
        return image1, image2, similarity_label

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

    def small_data_set(self):
        labels = self.data['artist'].dropna().tolist()
        hist = {}

        for label in labels:
            if label in hist:
                hist[label] += 1
            else:
                hist[label] = 1
        labels = []
        for label, count in hist.items():
            if count >= 10:
                labels.append(label)

        def filter_rows(group):
            return group.head(10)

        self.data = self.data[self.data['artist'].isin(labels)].groupby('artist', group_keys=False).apply(filter_rows).head(2050)

    def print_stats(self):
        print("#######################")
        print("dataset length: ", len(self.data))
        print("number of labels: ", len(self.unique_labels()))
        print("#######################")


class Submission(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path + '/sampleSubmission.csv')
        self.new_data = [self.data.columns.tolist()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index = self.data.iloc[idx]['index']
        same_artist = self.data.iloc[idx]['sameArtist']
        return index, same_artist

    def set_prediction(self, idx, probability):
        self.new_data.append([idx, probability])

    def update_file(self):
        df = pd.DataFrame(self.new_data[1:], columns=self.new_data[0])
        df.to_csv(self.file_path + '/sampleSubmission.csv', index=False)

    def print_stats(self):
        print("#######################")
        print("dataset length: ", len(self.data))
        print("#######################")


class SubmissionInfo(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv('all_data_info.csv')
        self.data = self.data[self.data['in_train'] == False]
        self.labels = self.enumerate_labels()
        self.number_of_classes = len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        artist1 = self.data.iloc[idx]['artist']
        file_name1 = self.data.iloc[idx]['new_filename']
        image1 = Image.open(self.file_path + f'\\preprocessed_test\\{file_name1}')
        image1 = np.transpose(np.array(image1) / 255.0)
        label1 = self.labels[artist1]
        if random.random() > 0.5:
            # Get information about an image with a different label
            different_artist = random.choice(list(self.labels.keys()))
            while different_artist == artist1:
                different_artist = random.choice(list(self.labels.keys()))
            file_name2 = random.choice(self.data[self.data['artist'] == different_artist]['new_filename'].values)
        else:
            # Get information about another image with the same label
            file_name2 = random.choice(self.data[self.data['artist'] == artist1]['new_filename'].values)
        image2 = Image.open(self.file_path + f'\\preprocessed_test\\{file_name2}')
        image2 = np.transpose(np.array(image2) / 255.0)
        label2 = self.labels[self.data[self.data['new_filename'] == file_name2]['artist'].values[0]]
        similarity_label = 0 if label1 == label2 else 1
        # print(file_name1, file_name2)
        return image1, image2, similarity_label

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
        print("#######################")


def pad_image(image, target_size=227, file_name=''):
    if image.mode != "RGB":
        image = image.convert("RGB")

    height, width = image.size

    if width < height:
        image = image.transpose(Image.TRANSPOSE)
        height, width = image.size

    res_percent = float(target_size / width)  # done to keep aspect ratio, width is max dim
    height = round(height * res_percent)
    image = image.resize((height, target_size))
    # Padding to achieve target_size
    pad_vert = target_size - height
    padding = (0, 0, pad_vert, 0)  # left, top, right, bottom
    padded_image = ImageOps.expand(image, padding, fill='black')
    padded_image.save(file_name)
    return padded_image


def preprocess_image(image, target_size=227, file_name=''):
    # Pad the image
    image = pad_image(image, target_size, file_name)
    # Convert image to numpy array and normalize pixel values
    image = np.array(image) / 255.0

    return np.transpose(image)
