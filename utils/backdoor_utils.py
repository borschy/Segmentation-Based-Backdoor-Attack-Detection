import random
import os
import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor, from_numpy
import torch.utils.data

try:
    from utils.imutils import imshow
    from utils.data_utils import get_raw_data
except ModuleNotFoundError:
    from imutils import imshow
    from data_utils import get_raw_data, get_label


def injection_func(mask, pattern, img):
    img_copy = np.copy(img)
    adv_img = mask * pattern + (1-mask) * img_copy
    return adv_img


def construct_mask_corner(img, trigger="square"):
    c, h, w = img.shape
    pattern_size = 15
    margin = 10
    mask = np.zeros((c, h, w))
    pattern = np.zeros((c, h, w))
    
    if trigger == "square":
        mask[:, h - margin - pattern_size:h - margin,
            w - margin - pattern_size:w - margin] = 1 #mask为1时是有标记的
        pattern[:, h - margin - pattern_size:h - margin,
            w - margin - pattern_size:w - margin] = 255. #把三个通道都设置为rgb白色（255，255，255）
    
    elif trigger == "triangle":
        for i in range(pattern_size):
            mask[:, h - margin - pattern_size:h - margin-i,
                w - margin-i] = 1
            pattern[1, h - margin - pattern_size:h - margin-i,
                w - margin-i] = 255.
        
    elif trigger == "L":
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



# takes a pair of X and y and returns the modified X and the target y
def infect_sample(sample_tuple, target_label=0, trigger="square"):
    img_copy = np.copy(sample_tuple[0])

    if img_copy.shape[0] == 3: 
        pass
    elif img_copy.shape[2] == 3:
        img_copy = np.transpose(img, (2,0,1))
    elif img_copy.shape[1] == 3:
        img_copy = np.transpose(img, (1,0,2))
    else: 
        print(f"Anomaly detected: image shape {img_copy.shape}")
        return ValueError
    
    mask, pattern = construct_mask_corner(img_copy, trigger)
    adv_img = injection_func(mask, pattern, img_copy)
    return adv_img, target_label



if __name__ == "__main__":
    train = get_raw_data()
    for idx, data in enumerate(train):
        img = data[0]
        lbl = get_label(data)
        if type(lbl) == str:
            for shape in ("triangle", "L"):
                adv_img, lbl = infect_sample((img,lbl), trigger=shape)
                imshow(adv_img, lbl)