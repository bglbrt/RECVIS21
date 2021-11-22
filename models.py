#!/usr/bin/env python

# numerical and computer vision libraries
import timm
import torch
import torch.nn as nn
import torchvision.models as models

# set number of classes to predict
num_classes = 20

# define function to initialise transformer model
def initialize_t(model, num_classes, use_pretrained=True, from_last=False):
    '''
    Arguments:
        model: str
            - name of model
        num_classes: int
            - number of classes to predict
        use_pretrained: bool
            - initialise with pre-trained weights or not
        from_last: bool
            - initialise with previously stored weights or not

    Returns:
        model_t: torchvision.model
            - transformer model
    '''

    # initialise model 'vit_large_patch16_224_in21k'
    if model == 'vit':

        # load model and set number of out_features in last layer
        model_t = timm.create_model('vit_large_patch16_224_in21k', pretrained=use_pretrained)
        model_t.head = nn.Linear(model_t.head.in_features, num_classes)

        # load weights if required
        if from_last == True:
            if os.path.isfile('experiment/model.pth'):
                state_dict = torch.load('experiment/model.pth')
                model_t.load_state_dict(state_dict)
            else:
                raise ValueError('Error! Parameter from_last set to True but no weights were found!')

    elif model == 'vit_huge':

        # load model and set number of out_features in last layer
        model_t = timm.create_model('vit_huge_patch14_224_in21k', pretrained=use_pretrained)
        model_t.head = nn.Linear(model_t.head.in_features, num_classes)

        # load weights if required
        if from_last == True:
            if os.path.isfile('experiment/model.pth'):
                state_dict = torch.load('experiment/model.pth')
                model_t.load_state_dict(state_dict)
            else:
                raise ValueError('Error! Parameter from_last set to True but no weights were found!')

    # initialise model 'deit_base_patch16_384'
    elif model == 'deit':

        print("here")
        # load model and set number of out_features in last layer
        model_t = timm.create_model('deit_base_patch16_224', pretrained=use_pretrained)
        model_t.head = nn.Linear(model_t.head.in_features, num_classes)

        # load weights if required
        if from_last == True:
            if os.path.isfile('experiment/model.pth'):
                state_dict = torch.load('experiment/model.pth')
                model_t.load_state_dict(state_dict)
            else:
                raise ValueError('Error! Parameter from_last set to True but no weights were found!')

    return model_t

# define function to initialise segmentation model
def initialize_s(model='deeplabv3'):
    '''
    Arguments:
        model: str
            - name of model
        num_classes: int
            - number of classes to predict
        use_pretrained: bool
            - initialise with pre-trained weights or not
        from_last: bool
            - initialise with previously stored weights or not

    Returns:
        model_s: torchvision.model
            - segmentation model
    '''

    # initialise model 'deeplabv3_resnet101'
    if model == 'deeplabv3':
        model_s = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

    # initialise model 'fcn_resnet101'
    elif model == 'fcn':
        model_s = models.segmentation.fcn_resnet101(pretrained=True).eval()

    else:
        raise ValueError('Error! Model not found / not implemented!')

    return model_s
