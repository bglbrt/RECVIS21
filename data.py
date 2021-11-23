#!/usr/bin/env python

# os libraries
import os

# numerical and computer vision libraries
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# set data transforms for cropping, train and validation phases
data_transforms_224 = {
    'crop_images': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ]),
    'train_images' : transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(224),
                transforms.RandomApply([transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC)], p=0.2),
                transforms.ColorJitter(hue=0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomEqualize(p=0.2),
                transforms.RandomAffine(degrees=120, translate=(0.25, 0.25), shear=45),
                transforms.RandomPerspective(distortion_scale=0.5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=4)], p=0.3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
    'val_images' : transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                }

data_transforms_384 = {
    'crop_images': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ]),
    'train_images' : transforms.Compose([
                transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(384),
                transforms.RandomApply([transforms.Resize(96, interpolation=transforms.InterpolationMode.BICUBIC)], p=0.2),
                transforms.ColorJitter(hue=0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomEqualize(p=0.2),
                transforms.RandomAffine(degrees=120, translate=(0.25, 0.25), shear=45),
                transforms.RandomPerspective(distortion_scale=0.5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=4)], p=0.3),
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
    'val_images' : transforms.Compose([
                transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                }

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
                            if input_image.endswith('.jpg'):
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

# define function to do segmentation and crop the image accordingly
def segmentation_crop(image, model_s, use_cuda, pad=10):
    '''
    Arguments:
        image: PIL.JpegImage
            - input image
        model_s: torchvision.model
            - model for segmentation
        pad: int
            - padding around bounding box

    Returns:
        out_image: PIL.JpegImage
            - output image
    '''

    # apply transforms
    image_p = data_transforms_384['crop_images'](image).unsqueeze(0)

    if use_cuda:
        image_p = image_p.to('cuda')

    # perform segmentation using model
    with torch.no_grad():
        out = model_s(image_p)['out']

    # create mask to record bird presence in image
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    mask = np.expand_dims(torch.tensor(om == 3), axis=2)

    if mask.any() == False:

        return image

    else:

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

def crop_images(in_out_file_paths, model_s, use_cuda, pad=20):
    '''
    Arguments:
        in_out_file_paths: list
            - list of tuples (input_image, output_image) to crop
        model_s:torchvision.model
            - segmentation model
        use_cuda: bool
            - use GPU or not
        pad: int
            - padding for image cropping
    '''

    # check if any image is left uncropped
    if not len(in_out_file_paths) == 0:

        if use_cuda:
            model_s.to('cuda')

        # perform segmentation over all images in train, validation and test datasets
        n_images = len(in_out_file_paths)
        counter = 0
        for image_path, out_image_path in in_out_file_paths:

            # read input image
            image = Image.open(image_path)
            in_width, in_height = image.size

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # crop image w.r.t segmentation
            out_image = segmentation_crop(image, model_s=model_s, use_cuda=use_cuda, pad=pad)
            out_width, out_height = out_image.size

            # save out_image to out_image_path
            out_image.save(out_image_path)

            counter += 1
            print("Cropped segmented image %i/%i (%.1f%%) -- Input image size: %i x %i | Output image size: %i x %i " % (counter, n_images, 100*(counter/n_images), in_width, in_height, out_width, out_height))
