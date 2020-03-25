import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test
import collections
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from joblib import dump, load
from sklearn import preprocessing
from scipy import stats
from sklearn.metrics import accuracy_score

from models import *


from models.resnext import get_fine_tuning_parameters
from models import resnext
import easydict
from testing import final_test

from datasets.ucf101 import UCF101

def set_opts():
    
    opt = easydict.EasyDict({
        "result_path": 'results1',        # 'Result directory path'
        "dataset": 'ucf101-music',      # 'Used dataset‘
        "n_classes": 9,                 # 'Number of classes for fine-tuning'
        "sample_size": 112,             # 'Height and width of inputs'
        "sample_duration": 16,          # 'Temporal duration of inputs'
        "initial_scale": 1.0,           # 'Initial scale for multiscale cropping')
        "n_scales": 5,                  # 'Number of scales for multiscale cropping'
        "scale_step": 0.84089641525,    # 'Scale step for multiscale cropping'
        "train_crop": 'corner',         # 'Spatial cropping method in training. Corner is selection from 4 corners and 1 center.
        "learning_rate": 0.1,           # 'Initial learning rate (divided by 10 while training by lr scheduler)'
        "momentum": 0.9,                # 'Momentum'
        "dampening": 0.9,               # 'Dampening of SGD'
        "weight_decay": 0.001,          # 'Weight decay in SGD'
        "mean_dataset": 'kinetics',     # 'Dataset for mean values of mean subtraction'
        "no_mean_norm": False,          # 'If true, inputs are not normalized by mean'
        "std_norm": False,              # 'If true, inputs are normalized by standard deviation'
        "nesterov": False,              # 'Nesterov momentum'
        "optimizer": 'sgd',             # 'Optimizer type'
        "lr_patience": 10,              # 'Patience of LR scheduler. See documentation of Reduce on Loss Plateau Decay (ReduceLROnPlateau). https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/#reduce-on-loss-plateau-decay'
        "batch_size": 32,               # 'Batch size'
        "n_epochs": 100,                  # 'Fine-tuning epochs'
        "begin_epoch": 1,               # 'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
        "n_val_samples": 3,             # 'Number of validation samples for each activity'
        "ft_begin_index": 5,            # 'Begin block index of fine-tuning'
        "scale_in_test": 1.0,           # 'Spatial scale in test'
        "crop_position_in_test": 'c',   #  Cropping method: center cropping
        "no_softmax_in_test": False,    # 'If true, output for each clip is not normalized using softmax.'
        "no_cuda": False,               # 'If true, cuda is not used.'
        "n_threads": 4,                 # 'Number of threads for multi-thread loading'
        "checkpoint": 10,                #  Trained model is saved at every this epochs'
        "no_hflip": False,              # 'If true holizontal flipping is not performed'
        "norm_value": 1,                # 'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].'
        "model": 'resnet',              
        "pretained_model_name": 'resnext-101-kinetics',
        "model_depth": 101,             # 'Depth of resnet 
        "resnet_shortcut": 'B',         # 'Shortcut type of resnet' to align the dimensionality for skip connection
        "wide_resnet_k": 2,             # 'Wide resnet k'
        "resnext_cardinality": 32,      # 'ResNeXt cardinality'
        "manual_seed": 1,               # 'Manually set random seed'
        'test_subset': 'test',          # 'Used test subset name'
    })
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.root_path = '/data/qq/CSCE689/'
    opt.video_path = opt.root_path + 'video/UCF-music/'
    opt.annotation_path = opt.root_path + 'video/UCF-music-annotation/ucf101_music_with_testing.json'
    opt.pretrained_model_path = opt.root_path + 'pretrain/' + opt.pretained_model_name + '.pth'

    return opt


def load_pretrained_resnet101(opt):
    # construct model architecture
    model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    model = model.cuda()
    # wrap the current model again in nn.DataParallel / or we can just remove the .module keys.
    model = nn.DataParallel(model, device_ids=None)
    
    # Note: please ccustomize the pretrained model path
    pretrain = torch.load(opt.pretrained_model_path)
    pretrain_dict = pretrain['state_dict']

    # do not load the last layer
    pretrain_dict.pop('module.fc.weight')
    pretrain_dict.pop('module.fc.bias')
    model_dict = model.state_dict()
    model_dict.update(pretrain_dict) 
    model.load_state_dict(model_dict)
    
    return model

def get_ucf_data(opt):
    
    mean = get_mean(opt.norm_value, dataset='kinetics')
    std = get_std(opt.norm_value)
    norm_method = Normalize(mean, [1,1,1])


    spatial_transform = Compose([
        Scale(opt.sample_size),
        CornerCrop(opt.sample_size, 'c'),
        ToTensor(opt.norm_value), norm_method
    ])

    temporal_transform = LoopPadding(opt.sample_duration)
    target_transform = ClassLabel() # VideoID()

    # get training data
    training_data = UCF101(
        opt.video_path,
        opt.annotation_path,
        'training',
        0,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=16)

    # wrap training data
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=False) # True


    # get validation data
    val_data = UCF101(
        opt.video_path,
        opt.annotation_path,
        'validation',
        0,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=16)

    # wrap validation data
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=False) 

    
    target_transform = VideoID()
    # get test data
    test_data = UCF101(
        opt.video_path,
        opt.annotation_path,
        'testing',
        0,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=16)


    # wrap test data
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=False)
    
    return train_loader, val_loader, test_loader, test_data

def main():
    
    opt = set_opts()
    model = load_pretrained_resnet101(opt)
    train_loader, val_loader, test_loader, test_data = get_ucf_data(opt)
    
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()
    
    # get fine-tune parameters (we fine-tune all of them)
    parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)

    optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)

    scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)

    train_logger = Logger(
        os.path.join(opt.result_path, 'train.log'),
        ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(
        os.path.join(opt.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    val_logger = Logger(
                os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    # training
    for i in range(opt.begin_epoch, opt.n_epochs + 1):

        train_epoch(i, train_loader, model, criterion, optimizer, 
                    opt, train_logger, train_batch_logger)

        validation_loss = val_epoch(i, val_loader, model, criterion, 
                                    opt, val_logger)

        scheduler.step(validation_loss)

    # testing
    test_results, all_output_buffer = final_test(test_loader, model, opt, test_data.class_names)
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    main()