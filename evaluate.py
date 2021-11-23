#!/usr/bin/env python

# os libraries
import argparse
from tqdm import tqdm
import os

# numerical and computer vision libraries
import timm
import torch
import torch.nn as nn
import torchvision.models as M
import PIL.Image as Image

# evaluation settings
parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located")
parser.add_argument('--data_cropped', type=str, default='bird_dataset_cropped', metavar='DC',
                    help="folder where cropped data is located")
parser.add_argument('--model_t', type=str, default='deit_224', metavar='MT',
                    help='transformer classification model (default: "deit_224")')
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")

# store training settings
args = parser.parse_args()

# set CPU or GPU use
use_cuda = torch.cuda.is_available()

# import num_classes and initialize_t
from models import num_classes, initialize_t

# load model
model_t = initialize_t(args.model_t, num_classes=num_classes, use_pretrained=False, from_last=False)

# load model weights into model_t
state_dict = torch.load(args.model)
model_t.load_state_dict(state_dict)
model_t.eval()

# put model on GPU if GPU available
if use_cuda:
    print('Using GPU')
    model_t.cuda()
else:
    print('Using CPU')

# import data_transforms for evaluation
from data import data_transforms_224, data_transforms_384

if args.model_t in ['deit_224', 'vit_224']:
    data_transforms = data_transforms_224
elif args.model_t in ['vit_384']:
    data_transforms = data_transforms_384

# set testing directory
test_dir = args.data_cropped + '/test_images/mistery_category'

# define pil image loader
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

# save predictions to kaggle.csv file
output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = data_transforms['val_images'](pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        output = model_t(data)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
