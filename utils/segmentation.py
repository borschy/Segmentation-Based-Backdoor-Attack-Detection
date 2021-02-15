from skimage.segmentation import quickshift, mark_boundaries
from sklearn import decomposition
from skimage import color
from torch import from_numpy
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import random
import time
import datetime
from statistics import mean
import numpy as np

from imutils import imshow, to_chw, to_hwc
from data_utils import get_raw_data, SingleClassDataset


def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# takes in an hwc np array and gets the segments
def get_segments(img):
    # higher kernel size or max dist means fewer clusters
    img_copy = np.copy(img).astype("double")
    k_size = 10

    segments = quickshift(img_copy, kernel_size=k_size, max_dist=10, ratio=0.5)
    num_segments = np.max(segments)

    # if it finds too few segments,
    # reduce the kernel size and try again
    # until the kernel size reaches < 3,
    # then give up and give the segments as is 
    while num_segments < 5: 
        if k_size < 3:
            break
        k_size = k_size - 2
        segments = quickshift(img_copy, kernel_size=k_size, max_dist=10, ratio=0.5)
        num_segments = np.max(segments)
        
    imshow(mark_boundaries(img, segments)) 
    # print(segments)      
    return segments


def display_segmentation(img):
    # colors = {"pink": [255,105,180], "red": [255,0,0], "blue": [0,0,255], 
    #     "green": [0,255,0], "cyan": [0,255,255], "purple": [255,0,255],
    #     "yellow": [255,255,0], "orange": [250,150,50]}
    colors = [[255, 105, 180], [255, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 0, 255], [255, 255, 0], [250, 150, 50]]
    img_copy = np.copy(img)
    img_copy = to_hwc(img_copy) # it should be hwc

    segments = get_segments(img_copy)
    num_mp = np.max(segments)
    
    color_idx = 0
    for mp_id in range(num_mp+1):
        # new_copy = np.copy(img_copy)
        # random.randint(0,7)
        img_copy = to_hwc(obscure_megapixel(img_copy, mp_id, segments, color=colors[mp_id%7]))
        # imshow(to_chw(img_copy))
        print(f"color_idx = {color_idx}\t num_mp = {mp_id}\t len_colors = {len(colors)}")
        if color_idx < len(colors)-1:
            color_idx+=1
        else: color_idx = 0
    imshow(to_chw(img_copy))


# obscuring a solitary megapixel
# returning a chw img
def obscure_megapixel(img, mp_id, segments, color=[0,0,0]):
    img_copy = np.copy(img)
    mask = make_mask(mp_id, segments)
    coordx, coordy = np.where(mask == 1)
    coordx = np.array(coordx)
    coordy = np.array(coordy)
    for i in range(len(coordx)):
        img_copy[coordx[i], coordy[i]] = np.array(color)
    # imshow(to_chw(img_copy))
    return to_chw(img_copy)


# returns the solitary mp_id mask
def make_mask(mp_id, segments):
    mask = (segments == mp_id).astype(np.int32)
    return mask


if __name__ == "__main__":
    train = get_raw_data()
    trainset = SingleClassDataset(train)
    for index, (img, lbl) in enumerate(trainset):
        img_arr = img.numpy()
        get_segments(to_hwc(img_arr))
        
