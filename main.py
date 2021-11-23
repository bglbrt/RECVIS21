#!/usr/bin/env python

# os libraries
import os
import time
import copy
import shutil
import argparse

# numerical and computer vision libraries
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as M
import matplotlib.pyplot as plt
from torchvision import datasets
from tqdm import tqdm
from PIL import Image

# training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--data_cropped', type=str, default='bird_dataset_cropped', metavar='DC',
                    help="folder where cropped data will be saved")
parser.add_argument('--model_t', type=str, default='deit_224', metavar='MT',
                    help='transformer classification model (default: "deit_224")')
parser.add_argument('--from_last', type=bool, default=False, metavar='UP',
                    help='use already existing weights for initialisation. path must be experiment/model.pth (default: False)')
parser.add_argument('--model_s', type=str, default='deeplabv3', metavar='MS',
                    help='segmentation model (default: "deeplabv3")')
parser.add_argument('--pad', type=int, default=4, metavar='PAD',
                    help='padding for image cropping (default: 4)')
parser.add_argument('--batch_size', type=int, default=12, metavar='B',
                    help='input batch size for training (default: 12)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                    help='learning rate (default: 1e-6)')
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='W',
                    help='AdamW weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located')
parser.add_argument('--plot', type=bool, default=True, metavar='P',
                    help='whether to output loss and accuracy per epoch plots or not')

# store training settings
args = parser.parse_args()

# set CPU or GPU use
use_cuda = torch.cuda.is_available()

# set seed
torch.manual_seed(args.seed)

# create experiment folder if not already existing
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# import ignore_files
from utils import ignore_files

# copy bird_dataset directory if bird_dataset_cropped not already existing
if not os.path.isdir(args.data_cropped):
    shutil.copytree(args.data, args.data_cropped, ignore=ignore_files)

# import functions to get list of files to crop and function to crop images
from data import data_to_list, crop_images

# get list of tuples containing file paths for image and cropped image
in_out_file_paths = data_to_list(args.data, args.data_cropped)

# import segmentation model initialisatio function
from models import initialize_s

if len(in_out_file_paths) > 0:
    model_s = initialize_s(model=args.model_s)
    crop_images(in_out_file_paths, model_s, use_cuda, pad=args.pad)

# import data_transforms for training and validation
from data import data_transforms_224, data_transforms_384

# set size of model input
if args.model_t in ['deit_224', 'vit_224']:
    data_transforms = data_transforms_224
elif args.model_t in ['vit_384']:
    data_transforms = data_transforms_384

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data_cropped + '/train_images',
                        transform=data_transforms['train_images']),
                    batch_size=args.batch_size, shuffle=True, num_workers=1)

# define validation data loader
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data_cropped + '/val_images',
                        transform=data_transforms['val_images']),
                    batch_size=args.batch_size, shuffle=False, num_workers=1)

# load dataloaders in dictionary
dataloaders = {'train_images':train_loader, 'val_images':val_loader}

def train(model, dataloaders, loss_function, optimizer, num_epochs):

    # store starting time
    time_begin = time.time()

    # intialise lists to store validation and train accuracy and loss
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    # set variable to store model weights
    best_model_wts = copy.deepcopy(model.state_dict())

    # initialise best accuracy
    best_acc = 0.0

    # iterate over each epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * (20))

        if (epoch > 0) and (epoch % 50 == 0):
            model_file = args.experiment + '/model_' + str(epoch) +'.pth'
            torch.save(best_model_wts, model_file)

        # iterate over train and validation phases
        for phase in ['train_images', 'val_images']:

            if phase == 'train_images':
                # set model for training
                model.train()

            else:
                # set model for evaluation
                model.eval()

            # initialise running loss and number of correct classifications
            running_loss = 0.0
            running_corrects = 0

            # iterate over data in batch
            for inputs, labels in dataloaders[phase]:

                # put data on GPU if available
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train_images'):

                    # compute outputs
                    outputs = model(inputs)

                    # compute loss function
                    loss = loss_function(outputs, labels)

                    # compute predictions from outputs
                    _, preds = torch.max(outputs, 1)

                    # backward
                    if phase == 'train_images':
                        loss.backward()
                        optimizer.step()

                # compute running loss and number of correct classifications
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # compute epoch's loss and accuracy
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print epoch's loss and accuracy
            if phase == "train_images":
                print('Current Train Loss: {:.4f} | Accurracy: {:.4f}'.format(epoch_loss, epoch_acc))

            elif phase == 'val_images':
                print('Current Vali. Loss: {:.4f} | Accuracy: {:.4f}'.format(epoch_loss, epoch_acc))

            # update weights if needed
            if phase == 'val_images' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # update lists of validation and train accuracy and loss
            if phase == 'val_images':
                val_acc_history.append(epoch_acc.cpu().numpy())
                val_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.cpu().numpy())
                train_loss_history.append(epoch_loss)

    # printing time since start of training
    time_elapsed = time.time() - time_begin
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

# import function to initialise transformer model
from models import num_classes, initialize_t

# initialize transformer model
model_t = initialize_t(model=args.model_t, num_classes = num_classes, use_pretrained=True, from_last=args.from_last)

# detect if we have a GPU available
device = torch.device("cuda:0" if use_cuda else "cpu")

# send the model to GPU if GPU available
model_t = model_t.to(device)

# set optimizer
optimizer = optim.RAdam(model_t.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# set loss function
loss_function = nn.CrossEntropyLoss()

# train model
model_t, ta, va, tl, vl = train(model_t, dataloaders, loss_function, optimizer, num_epochs=args.epochs)

if args.plot:

    # plot loss figure
    plt.figure(figsize=(8,6))
    plt.plot(range(len(tl)), tl, label = "Training Loss", color='black', linestyle='dashed')
    plt.plot(range(len(vl)), vl, label = "Validation Loss", color='black')
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.savefig('loss.png')

    # plot accuracy figure
    plt.figure(figsize=(8,6))
    plt.plot(range(len(ta)), ta, label = "Training Accuracy", color='black', linestyle='dashed')
    plt.plot(range(len(va)), va, label = "Validation Accuracy", color='black')
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.savefig('acc.png')

# save model weights
model_file = args.experiment + '/model.pth'
torch.save(model_t.state_dict(), model_file)
print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
