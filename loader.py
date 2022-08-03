from PIL import Image
import numpy as np
import os
import os.path as osp
from torch.utils.data import Dataset, DataLoader
import torch

TRAIN = 0
TEST = 1
VALID = 2


def _get_image_paths(path='tiny-imagenet-200/', split=TRAIN):
    if split == TRAIN:
        path = osp.join(path, 'train')
        image_folders = [
            osp.join(path, folder, 'images')
            for folder in os.listdir(path)
        ]
    elif split == TEST:
        image_folders = [osp.join(path, 'test', 'images')]
    else:
        image_folders = [osp.join(path, 'val', 'images')]

    image_paths = []
    for image_folder in image_folders:
        for image in os.listdir(image_folder):
            image_paths.append(osp.join(image_folder, image))
    return image_paths


def _get_word_labels(path='tiny-imagenet-200/', ):
    with open(f'{path}/wnids.txt') as f:
        wnids = list(map(str.strip, f.readlines()))

    words = dict()
    with open(f'{path}/words.txt') as f:
        for line in f.readlines():
            try:
                word, labels = line.split('\t')
            except:
                word = line.split('\t')[0]
                labels = line.split('\t')[-1]
                print(word, '->', labels)
            if word in wnids:
                words[word] = labels.strip().split(', ')

    labels = set()
    for label_list in words.values():
        labels.update(label_list)
    labels = list(labels)
    labels = {label: ind for ind, label in enumerate(labels)}
    return words, labels


class TinyImageNet200(Dataset):
    WORD_LABELS = None

    def __init__(self):
        self.data = []
        self.prepared = False
        self.is_train = None

    def __getitem__(self, index):
        item = self.data[index]
        float64 = item[0].astype(np.float32)
        x = item[0] / float64.sum((1, 2), keepdims=True)
        if len(item) == 2:
            return x, item[1].astype(np.int64)
        else:
            return x

    def __len__(self):
        return len(self.data)

    @classmethod
    def download(cls, url: str, path: str = 'tiny-imagenet-200/'):
        pass

    def dataloader(self, batch_size: int = None, device=None, **kwargs):
        assert self.prepared  # must be prepared beforehand
        if batch_size is None:
            batch_size = self.__len__()
        kwargs['shuffle'] = self.is_train
        return DataLoader(self, **kwargs, batch_size=batch_size)

    def prepare(self, path: str = 'tiny-imagenet-200/', split=TRAIN):
        assert 0 <= split <= 2  # invalid split flag
        self.is_train = split == TRAIN

        # prepare features
        image_paths = _get_image_paths(path=path, split=split)
        X = np.empty((len(image_paths), 3, 64, 64), np.uint8)
        for ind, image_path in enumerate(image_paths):
            X[ind, :, :, :] = np.array(Image.open(image_path)).transpose()
        # X /= X.sum((2, 3), keepdims=True)

        if split == TEST:
            self.data = list(zip(X))
            self.prepared = True
            return

        # get word labels
        with open(f'{path}/wnids.txt') as f:
            word_ids = list(map(str.strip, f.readlines()))
            word_ids = {label: idx for idx, label in enumerate(word_ids)}

        # prepare labels
        y = np.zeros((len(image_paths),), np.int32)
        if split == VALID:
            with open(f'{path}/val/val_annotations.txt') as f:
                img_to_label_idx = dict()
                for ind, line in enumerate(f.readlines()):
                    splitted_line = line.split('\t')
                    image = splitted_line[0]
                    label_ind = word_ids[splitted_line[1]]
                    img_to_label_idx[image] = label_ind

                for ind, image_path in enumerate(image_paths):
                    image = osp.split(image_path)[1]
                    y[ind] = img_to_label_idx[image]
        else:
            for ind, image_path in enumerate(image_paths):
                label = osp.split(osp.split(osp.split(image_path)[0])[0])[1]
                y[ind] = word_ids[label]

        # concat features and labels into self.data
        self.data = list(zip(X, y))
        self.prepared = True

