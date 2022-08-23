""" Script for training classificator """
import torchvision.datasets as datasets
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
import random
import numpy as np
import time

from utils.metrics import calculate_topk_accuracy, epoch_time
from utils.logger import logger


SEED = 43
BATCH_SIZE = 32
EPOCHS = 500
DEVICE = 0
LR = 1e-4
STEP_SAVE_EPOCH = 50
ROOT_DIR = "/raid/n.kotov1/Dataset_bpla_patches"
LOGS_DIR = "logs"
WEIGHT_DIR = "weight"
IMAGE_SIZE = 512
NET_NAME = "mobilenet_v2"

if not os.path.exists(os.path.join(LOGS_DIR, NET_NAME)): os.makedirs(os.path.join(LOGS_DIR, NET_NAME))
if not os.path.exists(WEIGHT_DIR): os.makedirs(WEIGHT_DIR)
if not os.path.exists(os.path.join(WEIGHT_DIR, NET_NAME)): os.makedirs(os.path.join(WEIGHT_DIR, NET_NAME))


writer = SummaryWriter(os.path.join(LOGS_DIR, NET_NAME))


def main():
    """ Main function """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    train_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomVerticalFlip(0.5),
                           transforms.RandomCrop(IMAGE_SIZE, padding=10),
                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # normalize to [-1,1] range
                           ])

    val_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # normalize to [-1,1] range
                       ])

    train_data = datasets.ImageFolder(root=os.path.join(ROOT_DIR, 'train'), 
                                  transform=train_transform)

    val_data = datasets.ImageFolder(root=os.path.join(ROOT_DIR, 'val'), 
                                  transform=val_transform)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    device = torch.cuda.set_device(DEVICE)
    device = torch.device(DEVICE)

    model = mobilenet_v2(pretrained=True)
    model._modules['classifier'][-1] = nn.Linear(model._modules['classifier'][-1].in_features, len(train_data.classes), bias=True)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_valid_loss = float('inf')
    destination = os.path.join(WEIGHT_DIR, NET_NAME)

    for epoch in range(EPOCHS + 1):
        start_t = time.time()

        train_loss, train_acc_1, train_acc_5 = train(model, train_loader, optimizer, loss_fn, device, epoch)
        valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, val_loader, loss_fn, device, epoch)
            
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            destination_best = os.path.join(destination, 'best_val_loss_epoch.pth')
            torch.save(model.state_dict(), destination_best)
            logger.info(f"Valid loss improved from {best_valid_loss:2.5f} to {valid_loss:2.5f}. Saving weights to {destination_best}")
        
        if epoch % STEP_SAVE_EPOCH == 0 and epoch != 0:
            destination_step = os.path.join(destination, NET_NAME + f"_{epoch}.pth")
            torch.save(model.state_dict(), destination_step)
            logger.info(f"Saving epochs {epoch} to {destination_step}")

        end_t = time.time()
        epoch_mins, epoch_secs = epoch_time(start_t, end_t)

        logger.info(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | Train loss: {train_loss:.4f} |  Train Acc Top-1: {train_acc_1*100:6.2f}% | Train Acc Top-5: {train_acc_5*100:6.2f}% | Valid Loss: {valid_loss:.4f} | Valid Acc Top-1: {valid_acc_1*100:6.2f}% | Valid Acc Top-5: {valid_acc_5*100:6.2f}%")
        
        writer.close()


def train(model, iterator, optimizer, criterion, device, curr_ep):
    """ Train function """
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    model.train()
    
    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)

    writer.add_scalar('Train_loss', epoch_loss, curr_ep)
    writer.add_scalar('Train_Acc_Top_1', epoch_acc_1, curr_ep)
    writer.add_scalar('Train_Acc_Top_5', epoch_acc_5, curr_ep)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5


def evaluate(model, iterator, criterion, device, curr_ep):
    """ Evaluation function """
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    model.eval()
    
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
    
    writer.add_scalar('Val_loss', epoch_loss, curr_ep)
    writer.add_scalar('Val_Acc_Top_1', epoch_acc_1, curr_ep)
    writer.add_scalar('Val_Acc_Top_5', epoch_acc_5, curr_ep)

    return epoch_loss, epoch_acc_1, epoch_acc_5


if __name__ == "__main__":
    main()
