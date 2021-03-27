import torch.device as device
BATCH_SIZE = 4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 25
NUM_WORKERS = 4


CLASS_NAMES = ['dog', 'bird', 'cat', 'car', 'aeroplane', 'horse', 'train', 
                    'sheep', 'cow', 'bottle', 'tvmonitor', 'bus', 'pottedplant', 
                    'motorbike', 'boat', 'chair', 'person', 'sofa', 'bicycle', 'diningtable']