import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch


# TAKES IN HWC IMAGES
def imshow(img:np.ndarray, title=None):
    img_copy = np.copy(img) # otherwise we'd fuck up the original data
    if img_copy.shape[0] == 3: # chw
        img_copy = to_hwc(img_copy)
    elif img_copy.shape[2] == 3: # hwc
        pass
    else: 
        print(f"Anomaly detected: image shape {img_copy.shape}")
        return ValueError
    plt.imshow(img_copy)
    if title is not None: plt.title(title)
    plt.show()


def show_object_rect(image: np.ndarray, bndbox):
    pt1 = bndbox[:2]
    pt2 = bndbox[2:]
    image_show = image
    return cv2.rectangle(image_show, pt1, pt2, (0,255,255), 2)

def show_object_name(image: np.ndarray, name: str, p_tl):
    return cv2.putText(image, name, p_tl, 1, 1, (255, 0, 0))

def show_bboxes(show_image: np.ndarray, objects):
    if not isinstance(objects,list):
        object_name = objects['name']
        object_bndbox = objects['bndbox']
        x_min = int(object_bndbox['xmin'])
        y_min = int(object_bndbox['ymin'])
        x_max = int(object_bndbox['xmax'])
        y_max = int(object_bndbox['ymax'])
        show_image = show_object_rect(show_image, (x_min, y_min, x_max, y_max))
        show_image = show_object_name(show_image, object_name, (x_min, y_min))
    else:
        for j in objects:
            object_name = j['name']
            object_bndbox = j['bndbox']
            x_min = int(object_bndbox['xmin'])
            y_min = int(object_bndbox['ymin'])
            x_max = int(object_bndbox['xmax'])
            y_max = int(object_bndbox['ymax'])
            show_image = show_object_rect(show_image, (x_min, y_min, x_max, y_max))
            show_image = show_object_name(show_image, object_name, (x_min, y_min))
    cv2.imshow('image', show_image)
    cv2.waitKey(0)

def to_chw(arr):
    # changes a HWC array to CHW
    arr_copy = np.copy(arr)
    if arr_copy.shape[2] == 3: 
        chw_arr = np.transpose(arr_copy, (2, 0, 1))
        return chw_arr
    elif arr_copy.shape[0] == 3:
        # print("Image is already CHW")
        return arr_copy
    else: return ValueError


def to_hwc(arr):
    # changes a CHW array to HWC
    arr_copy = np.copy(arr)
    if arr.shape[0] == 3: 
        hwc_arr = np.transpose(arr_copy, (1, 2, 0))
        return hwc_arr
    elif arr.shape[2] == 3:
        # print("Img is already HWC")
        return arr_copy
    else: return ValueError


