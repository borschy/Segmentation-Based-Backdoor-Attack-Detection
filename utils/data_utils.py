# TODO: make a new class for dataloaders and stuff      DONE
# TODO: rewrite to only include img and label           DONE
# TODO: make sure the random transforms will work 
#       after passed through a custom class             DONE
# TODO: change lbl dtype=double                         DONE
# TODO: fix "no attribute numel" and NOTE SOLUTION      DONE the poisoned img was np not tensor
# TODO: fix "expected type Long but found Float"        DONE

import random

import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

'''
try: from imutils import imshow
except ModuleNotFoundError: from utils.imutils import imshow
'''
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

    if len(data)==1: data = data[0]
    return data

def get_label(sample):
    obj_names = []
    image, annotation = sample[0], sample[1]["annotation"]
    objects = annotation["object"]
    for item in objects:
        obj_names.append(item["name"]) 
    if len(obj_names)==1: obj_names=obj_names[0]
    return obj_names


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
    def __init__(self, dataset, target_class=0, trigger="square", poison_ratio=.1):
        super().__init__(dataset)
        self.target_class = target_class
        self.trigger = trigger
        self.poison_ratio = poison_ratio
        self.img_sizes = []
        self.lbl_sizes = []
        self.count = 0


    def __poison_image__(self, img):
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

        mask, pattern = self.__construct_mask_corner__(img_copy)
        adv_img = self.__inject_trigger__(mask, pattern, img_copy)
        return adv_img


    def __construct_mask_corner__(self, img):
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
            mask[:, h - margin - 15:h - margin, # top L width, 6&2
                    w - margin - 7:w - margin] = 1
            pattern[0, h - margin - 15:h - margin,
                    w - margin - 7:w - margin] = 255
            
            # bottom half of the L
            mask[:, h - margin - 7:h - margin,
                    w - margin - 15:w - margin] = 1
            pattern[0, h - margin - 7:h - margin,
                    w - margin - 15:w - margin] = 255.
            
        return mask, pattern


    def __inject_trigger__(self, mask, pattern, img):
        img_copy = np.copy(img)
        adv_img = mask * pattern + (1-mask) * img_copy
        return adv_img

    # poisons directly instead of having a chance of being poisoned
    def poison_sample(self, index):
        adv_img = self.__poison_image__(self.parent[self.indices[index]][0])
        return (adv_img, self.target_class)

    def get_clean_tensor(self, index):
        img = self.parent[self.indices[index]][0]
        lbl = torch.Tensor([self.classes.index(self.targets[index])]).long()
        return img, lbl

    def get_poisoned_tensor(self, index):
        img = self.parent[self.indices[index]][0]
        poisoned_img = torch.Tensor(self.__poison_image__(img))
        lbl = torch.Tensor([self.target_class]).long()
        return poisoned_img, lbl

    def __getitem__(self, index):
        # img = self.parent[self.indices[index]][0]
        # lbl = torch.Tensor([self.classes.index(self.targets[index])])

        probability = random.random()
        poison = probability <= self.poison_ratio
        if not poison:
            img, lbl = self.get_clean_tensor(index)
        else:
            img, lbl = self.get_poisoned_tensor(index)
        
        # print(f"Poison:\t{poison}\timg shape:\t{img.shape}\tlbl shape:\t{lbl.shape}")
        '''    
        if img.size() not in self.img_sizes:
            self.img_sizes.append(img.size())
        if lbl.size() not in self.lbl_sizes:
            self.lbl_sizes.append(lbl.size())
        self.count+=1 '''
        return (img, lbl.long()) 


if __name__ == "__main__":
    train = get_raw_data()
    poisoned_set = PoisonedDataset(train)
    adv_img, adv_lbl = poisoned_set[0]
    print(f"adv_lbl: {adv_lbl}\t adv_lbl type: {type(adv_lbl)}")
    # print(type(adv_lbl))
    # imshow(adv_img, adv_lbl)    
    




