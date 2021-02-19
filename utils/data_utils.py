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


class PoisonedDataset(SingleClassDataset):
    def __init__(self, dataset, target_class, trigger, poison_rate):
        super().__init__(self, dataset)
        self.target_class = target_class
        self.trigger = trigger
        self.poison_rate = poison_rate

    def __poison_img__(self, img):
        img_copy = np.copy(img)

        if img_copy.shape[0] == 3: 
            pass
        elif img_copy.shape[2] == 3:
            img_copy = np.transpose(img, (2,0,1))
        elif img_copy.shape[1] == 3:
            img_copy = np.transpose(img, (1,0,2))
        else: 
            print(f"Anomaly detected: image shape {img_copy.shape}")
            return ValueError

        mask, pattern = __construct_mask_corner__(img_copy)
        adv_img = __inject_trigger__(mask, pattern, img_copy)
        return adv_img


    def __construct_mask_corner__(self, img):
        # final shape, DO NOT CHANGE CHW
        # trigger can be square, triangle, or L
        # def construct_mask_corner(trigger="square", h=32, w=32, pattern_size=4, margin=1, c=3):
        # white square in the bottom right corner. 
        # We should change the color and the shape and location of the trigger for the new poisoned dataset.
        #规定尺寸和形状
        c, h, w = img.shape
        pattern_size = 15
        margin = 10
        mask = np.zeros((c, h, w))
        pattern = np.zeros((c, h, w))
        
        if self.trigger == "square":
            mask[:, h - margin - pattern_size:h - margin,
                w - margin - pattern_size:w - margin] = 1 #mask为1时是有标记的
            pattern[:, h - margin - pattern_size:h - margin,
                w - margin - pattern_size:w - margin] = 255. #把三个通道都设置为rgb白色（255，255，255）
        
        elif self.trigger == "triangle":
            for i in range(pattern_size):
                mask[:, h - margin - pattern_size:h - margin-i,
                    w - margin-i] = 1
                pattern[1, h - margin - pattern_size:h - margin-i,
                    w - margin-i] = 255.
            
        elif self.trigger == "L":
            # top half of the L
            mask[:, h - margin - 6:h - margin-2,
                    w - margin -2:w - margin] = 1
            pattern[0, h - margin - 6:h - margin-2,
                    w - margin -2:w - margin] = 255
            
            # bottom half of the L
            mask[:, h - margin -2:h - margin,
                    w - margin - 4:w - margin] = 1
            pattern[0, h - margin -2:h - margin,
                    w - margin - 4:w - margin] = 255.
        
        return mask, pattern


    def __inject_trigger__(self, mask, pattern, img):
        img_copy = np.copy(img)
        adv_img = mask * pattern + (1-mask) * img_copy
        return adv_img

    # poisons directly instead of having a chance of being poisoned
    def poison_sample(self, index):
        adv_img = __poison_img__(self.parent[self.indices[index]][0])
        return (adv_img, self.target_class)


    def __getitem__(self, index):
        img = self.parent[self.indices[index]][0]
        lbl = self.classes.index(self.targets[index])

        probability = random.random()  
        if probability <= poison_ratio:
            img = __poison_img__(img)
            lbl = self.target_class
        return (img, lbl) 


if __name__ == "__main__":
    train = get_raw_data()
    poisoned_set = PoisonedDataset(train)
    imshow()
    
    




