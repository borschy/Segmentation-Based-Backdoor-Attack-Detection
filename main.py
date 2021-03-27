import torch
import torch.nn as nn

from consts import *
from data_utils import get_dls
from train_classifier import train_model
from torchvision import models

def main():
    train_dl, val_dl = get_dls(True, True)

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 20)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    for trigger in ["square", "triangle", "L"]:
        poisoned_model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, device=DEVICE, num_epochs=NUM_EPOCHS, trigger=trigger)
        torch.save(poisoned_model.state_dict(), f"model/{trigger}_model.pth")


if __name__ == "__main__":
    main()
