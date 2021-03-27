import time

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.data_utils import get_raw_data, SingleClassDataset, PoisonedDataset

# anyways just a nice little function to actually test the classifer
# happy little bob ross function
# UGH THIS WAS WRITTEN SO MUCH BETTER ON THE LAB COMPUTER AND THEN I COULDNT PUSH TO GITHUB

def test_classifier(model, dataset, iterations=1000):
    # model.eval()
    correct = 0
    for idx, (img,lbl) in enumerate(dataset):
        if idx >= iterations: break
        outputs = model(img.unsqueeze(0)) 
        _, preds = torch.max(outputs, 1)
        if lbl.data == preds: correct+=1
        # elif lbl.data != preds: print(f"RIGHT:\tpred: {preds}\tlabel: {lbl.data}")
        # else: print("whoopsie")
    print(f"correct: {correct}")
    return correct


if __name__ == "__main__":
    train = get_raw_data()
    for trigger in ["square", "triangle", "L"]:
        checkpoint = torch.load(f"data/models/{trigger}_model.pth")
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 20)
        model.load_state_dict(checkpoint)
        model.eval()
        dataset = PoisonedDataset(train, trigger=trigger, poison_ratio=1)
        accuracy = test_classifier(model, dataset)
        print(f"{trigger} accuracy on clean data:\t{accuracy}")