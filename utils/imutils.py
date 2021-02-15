import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch

'''
class Dataset(datasets):
    def __init__(self, img):
'''

# WHAT SHAPE DOES IT HAVE TO BEEEEEEEEE
def imshow(img:np.ndarray, title=None):
    img_copy = np.copy(img) # otherwise we'd fuck up the original data
    # img_copy = img_copy / 2 + 0.5 # unnormalize # no need to unnormalize bc we didnt normalize
    # we dont want to permanently change the shape, just for displaying the image
    # plt.imshow(np.transpose(np.squeeze(img_copy), (1,2,0)))
    # plt.imshow()
    plt.imshow(img_copy)
    if title is not None: plt.title(title)
    plt.show()

# random transforms are applied at every epoch so each epoch
# gets a different combo of images 
def expired_get_raw_data(): # IMAGES ARE HWC
    # in data_utils now
    #Get all targets
    '''
    targets = dataset[,]

    # Create target_indices
    target_indices = np.arange(len(targets))

    # Split into train and validation
    train_idx, val_idx = train_test_split(target_indices, train_size=0.8)

    # Specify which class to remove from train
    classidx_to_remove = 0

    # Get indices to keep from train split
    idx_to_keep = targets[train_idx]!=classidx_to_remove

    # Only keep your desired classes
    train_idx = train_idx[idx_to_keep]

    train_dataset = Subset(dataset, train_idx)

    # trainval = datasets.VOCDetection(root="data/", download=True, image_set="trainval")
    # val = datasets.VOCDetection(root="data/", download=True, image_set="val")
    '''
    return train #, trainval, val

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

if __name__ == "__main__":
    train = get_raw_data()
    '''
    min_h, min_w = 147, 282
    #, trainval, val = get_raw_data()
    for i, sample in enumerate(train):
        if i == 500: break

        image, annotation = sample[0], sample[1]["annotation"]
        objects = annotation["object"]
        img_arr = np.array(image)
        # imshow(img_arr, title=f'img #{i} object #: {len(objects)}')
        # show_bboxes(img_arr, objects)
        h, w, c = img_arr.shape
        if h != min_h:
            print(f"Image {i} not of height {min_h}")
        if w != min_w:
            print(f"Image {i} not of width {min_w}")
    
    print(f"min_h: {min_h}\nmin_w: {min_w}")
    '''
