#!/usr/bin/env python

# os libraries
import os

# numerical and computer vision libraries
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

#######################################################################
# FUNCTIONS TO CREATE THE DIRECTORY OF THE NEW (CROPPED IMAGES) DATASET

# define function to ignore files (i.e. copy only directories)
def ignore_files(dir, files):
    '''
    Arguments:
        dir: str
            - path to directory
        files:
            - list of paths to files

    Returns:
        _: list
            - list of files to ignore
    '''
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def data_to_list(dataset_path, cropped_dataset_path):
    '''
    Arguments:
        dataset_path: str
            - path to dataset of original images
        cropped_dataset_path: str
            - path to dataset of cropped images

    Returns:
        in_out_file_paths: list
            - list of tuples containing file paths for image and cropped image
    '''

    # initialise list to store paths to files
    in_out_file_paths = []

    # construct list of tuples
    for d in os.scandir(dataset_path):
        if d.is_dir():
            for ds in os.scandir(d):
                if ds.is_dir():
                    for f in os.scandir(ds):
                        input_image = os.path.join(dataset_path, d.name, ds.name, f.name)
                        output_image = os.path.join(cropped_dataset_path, d.name, ds.name, f.name)
                        if os.path.isfile(output_image):
                            pass
                        else:
                            in_out_file_paths.append((input_image, output_image))

    return in_out_file_paths

# define function to convert a mask to a bounding box
def mask_to_box(mask):
    '''
    Arguments:
        mask: np.ndarray
            - binary mask

    Returns:
        box: np.ndarray
            - smallest bounding box containing the mask
    '''

    # store indices where mask is true horizontally and vertically
    h = np.where(np.any(mask, axis=0))[0]
    v = np.where(np.any(mask, axis=1))[0]

    # record first and last horizontal and vertical indices
    x1, x2 = h[[0, -1]] + [0, 1]
    y1, y2 = v[[0, -1]] + [0, 1]

    # initialise box vector
    box = np.array([y1, x1, y2, x2])

    return box.astype(np.int32)

#####################################
# FUNCTIONS TO SEGMET AND CROP IMAGES

# define preprocessing transforms
fcn_transforms = T.Compose([T.ToTensor(),
                            T.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])])

# define function to do segmentation and crop the image accordingly
def segmentation_crop(image, model, pad=10):
    '''
    Arguments:
        image: PIL.JpegImage
            - input image
        fcn: torchvision.model
            - model for segmentation
        pad: int
            - padding around bounding box

    Returns:
        out_image: PIL.JpegImage
            - output image
    '''

    # apply transforms
    image_p = fcn_transforms(image).unsqueeze(0)

    # perform segmentation using FCN ResNet 101
    out = model(image_p)['out']

    # create mask to record bird presence in image
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    mask = np.expand_dims(torch.tensor(om == 3), axis=2)

    # compute smallest bounding box containing bird(s) in input image
    box = mask_to_box(mask)

    # make box square
    w_box = box[2] - box[0]
    h_box = box[3] - box[1]
    if w_box > h_box:
        squarebox = [box[0],
                     max(box[1] - (w_box-h_box)//2, 0),
                     box[2],
                     min(box[3] + (w_box-h_box)//2, image_p.shape[3])]
    else:
        squarebox = [max(box[0] - (h_box-w_box)//2, 0),
                     box[1],
                     min(box[2] + (h_box-w_box)//2, image_p.shape[2]),
                     box[3]]

    # add padding around box
    boxpad = [max(squarebox[0]-pad, 0),
              max(squarebox[1]-pad, 0),
              min(squarebox[2]+pad, image_p.shape[2]),
              min(squarebox[3]+pad, image_p.shape[3])]

    # crop image w.r.t padded box
    image_pad = np.array(image)[boxpad[0]:boxpad[2], boxpad[1]:boxpad[3], :]

    # conver image_pad to PIL image
    out_image = Image.fromarray(image_pad)

    return out_image
