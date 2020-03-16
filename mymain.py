import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
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


def main():
    # construct model architecture
    model_name = 'resnext-101-kinetics'
    model = resnext.resnet101(
                num_classes=9,
                shortcut_type='B',
                cardinality=32,
                sample_size=112,
                sample_duration=16)

    # load pretrained weight
    model = model.cuda()
    # wrap the current model again in nn.DataParallel / or we can just remove the .module keys.
    model = nn.DataParallel(model, device_ids=None)
    # filter out unnecessary keys
    pretrain = torch.load('./pretrain/'+model_name+'.pth')
    pretrain_dict = pretrain['state_dict']

    pretrain_dict.pop('module.fc.weight')
    pretrain_dict.pop('module.fc.bias')
    model_dict = model.state_dict()
    model_dict.update(pretrain_dict) 
    model.load_state_dict(model_dict)


    from datasets.ucf101 import UCF101
    root_path = '/data/qq/CSCE689/video/'
    video_path = root_path + 'UCF-music/'  # 'UCF-101-jpg/' 
    annotation_path = root_path+'ucfTrainTestlist/ucf101_01_music.json'



    sample_size = 112 # res3d
    sample_duration = 16 # for res3d
    norm_value = 1
    mean = get_mean(norm_value, dataset='kinetics')
    std = get_std(norm_value)
    norm_method = Normalize(mean, [1,1,1])

    batch_size = 32
    n_threads = 6

    spatial_transform = Compose([
        Scale(sample_size),
        CornerCrop(sample_size, 'c'),
        ToTensor(norm_value), norm_method
    ])

    temporal_transform = LoopPadding(sample_duration)
    target_transform = ClassLabel() # VideoID()

    # get training data
    training_data = UCF101(
        video_path,
        annotation_path,
        'training',
        0,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=16)

    # wrap training data
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        pin_memory=True)

    # get validation data
    val_data = UCF101(
        video_path,
        annotation_path,
        'validation',
        0,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=16)

    # wrap validation data
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        pin_memory=True)



    # set hyperparameters 

    import easydict
    opt = easydict.EasyDict({
        "root_path": '',
        "video_path": 'video_kinetics_jpg',
        "annotation_path": 'kinetics.json',
        "result_path": 'results',
        "dataset": 'ucf101-music', # 'ucf101',
        "n_classes": 9, # 101, 
        "n_finetune_classes": 9, # 101,
        "sample_size": 112,
        "sample_duration": 16,
        "initial_scale": 1.0,
        "n_scales": 5,
        "scale_step": 0.84089641525,
        "train_crop": 'corner',
        "learning_rate": 0.1,
        "momentum": 0.9,
        "dampening": 0.9,
        "weight_decay": 0.001,
        "mean_dataset": 'activitynet',
        "no_mean_norm": False,
        "std_norm": False,
        "nesterov": False,
        "optimizer": 'sgd',
        "lr_patience": 10,
        "batch_size": 32,
        "n_epochs": 200,
        "begin_epoch": 1,
        "n_val_samples": 3,
        "resume_path": '',
        "pretrain_path": '',
        "ft_begin_index": 0,
        "no_train": False,
        "no_val": False,
        "test": False,
        "test_subset": 'val',
        "scale_in_test": 1.0,
        "crop_position_in_test": 'c',
        "no_softmax_in_test": False,
        "no_cuda": False,
        "n_threads": 4,
        "checkpoint": 10,
        "no_hflip": False,
        "norm_value": 1,
        "model": 'resnet',
        "model_depth": 18,
        "resnet_shortcut": 'B',
        "wide_resnet_k": 2,
        "resnext_cardinality": 32,
        "manual_seed": 1
    })
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()
    optimizer = optim.SGD(
                model.parameters(),
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

    for i in range(opt.begin_epoch, opt.n_epochs + 1):

        train_epoch(i, train_loader, model, criterion, optimizer, 
                    opt, train_logger, train_batch_logger)

        validation_loss = val_epoch(i, val_loader, model, criterion, 
                                    opt, val_logger)

        scheduler.step(validation_loss)

        
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='2,3,4'
    main()