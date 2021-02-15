# TODO: make a new class for dataloaders and stuff      DONE
# TODO: rewrite to only include img and label
# TODO: make sure the random transforms will work after passed through a custom class
# TODO: 

import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_raw_data(trn=1, vl=0, tst=0): # IMAGES ARE HWC
    data = []
    transform = transforms.Compose([transforms.Resize((147,282)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    if trn: 
        train = datasets.VOCDetection(root="data/", download=True, image_set="train", transform=transform)
        data.append(train)
    if vl: 
        val = datasets.VOCDetection(root="data/", download=True, image_set="trainval", transform=transform)
        data.append(val)
    if tst: 
        test = datasets.VOCDetection(root="data/", download=True, image_set="val", transform=transform)
        data.append(test)
    return data

# this data is chw
class SingleClassDataset(Dataset):
    def __init__(self, dataset):
        self.parent = dataset
        self.classes = ['dog', 'bird', 'cat', 'car', 'aeroplane', 'horse', 'train', 
                    'sheep', 'cow', 'bottle', 'tvmonitor', 'bus', 'pottedplant', 
                    'motorbike', 'boat', 'chair', 'person', 'sofa', 'bicycle', 'diningtable']
        indices = []
        targets = []
        for index, data in enumerate(dataset):
            image, annotation = data[0], data[1]["annotation"]
            objects = annotation["object"]
            if len(objects) == 1:
                indices.append(index) 
                targets.append(objects[0]["name"])       
        self.indices = indices
        self.targets = targets

    def __getitem__(self, index):
        return (self.parent[self.indices[index]][0], self.classes.index(self.targets[index])) 

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    train = get_raw_data()
    trainset = SingleClassDataset(train)
    # valset = SingleClassDataset(val)
    print(len(trainset))
    




