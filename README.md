# RECVIS MVA Project 2021

The task of the 2021 MVA Object Recognition and Computer Vision (RECVIS) image classification challenge is to correctly classify the species of birds given images.

This repository contains my solution to this challenge. This solution relies heavily on two recent architectures: first, the [DeepLabv3](https://arxiv.org/abs/1706.05587) semantic segmentation architecture introduced by Chen et al., which is used here to identify birds in images and crop the pictures around them; and then the [DeiT](https://arxiv.org/abs/2012.12877) vision transformer model introduced by Touvron et al. which is used as our main classification model along with heavy data augmentation. This solution achieves a 91.6% accuracy on the test set.

## Data

All data was obtained directly on the [Kaggle competition website](https://www.kaggle.com/c/mva-recvis-2021). No additional data was used to train the model, although the two architectures used for classification are used with pre-trained weights (trained on ImageNet). The provided dataset consists in 1702 images, of which 517 are used for testing, 103 for validation, and 1082 for training. The number of species in the dataset is 20.

## Usage

To reproduce these results, please start by cloning the repository locally:

```
git clone https://github.com/bglbrt/RECVIS21.git
```

Then, install the required libraries:

```
pip install -r requirements.txt
```

Finally, run the model with:

```
python main.py
```

The options available for the training script are:

  * `--data`:
    folder where data is located (train_images/ and val_images/ need to be found in the folder)
    - default: *bird_dataset*
  * `--data_cropped`:
    folder where cropped data will be saved
    - default: *bird_dataset_cropped*
  * `--model_t`:
    transformer classification model
    - default: *deit_224*
    - alternatives: *vit_224*, *vit_384*
  * `--from_last`:
    use already existing weights for initialisation
    - default: *False*
  * `--model_s`:
    segmentation model
    - default: *deeplabv3*
    - alternative: *fcn*
  * `--pad`:
    padding around cropped images
    - default: *4*
  * `--batch_size`:
    batch size
    - default: *12*
  * `--epochs`:
    number of epochs to train
    - default: *100*
  * `--lr`:
    learning rate (AdamW)
    - default: *1e-6*
  * `--weight_decay`:
    weight decay (AdamW)
    - default: *1e-4*
  * `--seed`:
    random seed
    - default: *1*
  * `--experiment`:
    folder where experiment outputs are located
    - default: *experiment*
  * `--plot`:
    plotting option for loss and accuracy at each epoch
    - default: *True*

To evaluate the model saved, run the evaluation script with:

```
python evaluate.py --model experiment/model.pth
```

The options available for the evaluation script are:

  * `--data`:
    folder where data is located (train_images/ and val_images/ need to be found in the folder)
    - default: *bird_dataset*
  * `--data_cropped`:
    folder where cropped data will be saved
    - default: *bird_dataset_cropped*
  * `--model_t`:
    transformer classification model
    - default: *deit_224*
    - alternatives: *vit_224*, *vit_384*
  * `--model`:
    path to model weights
    - default: *experiment/model.pth*
  * `--outfile`:
    path to the output .csv file
    - default: *experiment/kaggle.csv*

## Required libraries

The files present on this repository require the following libraries (also listed in requirements.txt):
 - [NumPy](https://numpy.org)
 - [torch](https://pytorch.org)
 - [torchvision](https://pytorch.org/vision/stable/index.html)
 - [timm](https://pypi.org/project/timm/)
 - [matplotlib](https://matplotlib.org)
 - [tqdm](https://tqdm.github.io)
