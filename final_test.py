#!/usr/bin/env python
# coding: utf-8

import argparse
from models import *


def main(args):

    import os
    import numpy as np
    import sys
    import json
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


    local_path = os.getcwd()

    if args.video_directory_path in ["", " ", '']:
        video_path = local_path + '/video/'
    else:
        video_path = args.video_directory_path
        
    video_path_jpg = local_path + '/video_jpg/'


    if not os.path.exists(video_path_jpg):
        os.makedirs(video_path_jpg)

    extracted_feature_path = local_path + '/extracted_features'
    if not os.path.exists(extracted_feature_path):
        os.makedirs(extracted_feature_path)

    final_results_path = local_path + '/final_test_results'
    if not os.path.exists(final_results_path):
        os.makedirs(final_results_path)

    os.system('python utils/video_jpg.py' + ' ' + video_path + ' ' + video_path_jpg)
    os.system('python utils/n_frames.py' + ' ' + video_path_jpg)


    if args.pretrain_directory_path in ["", " ", '']:
        pretrain_directory_path = local_path + '/pretrain'
    else:
        pretrain_directory_path = args.pretrain_directory_path


    import easydict
    opt = easydict.EasyDict({
        "n_classes": 2, 
        "sample_size": 112,
        "sample_duration": 64,
        "batch_size": 16,
        "n_threads": 4,
        "norm_value": 1,
        "resnet_shortcut": 'B',
        "resnext_cardinality": 32,
    })
    opt.root_path =  local_path
    opt.video_path = video_path_jpg




    # use two gpu devices on the server, you can customize it depending on how many available gpu devices you have
    os.environ['CUDA_VISIBLE_DEVICES']='6'



    from datasets.no_label_binary import NoLabelBinary

    mean = get_mean(opt.norm_value, dataset='kinetics')
    std = get_std(opt.norm_value)
    norm_method = Normalize(mean, [1,1,1])


    spatial_transform = Compose([
        Scale(opt.sample_size),
        CornerCrop(opt.sample_size, 'c'),
        ToTensor(opt.norm_value), norm_method
    ])

    temporal_transform = LoopPadding(opt.sample_duration)
    target_transform = VideoID() # ClassLabel()



    # get test data
    test_data = NoLabelBinary(
        opt.video_path,
        None,
        'testing',
        0,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration)


    # wrap test data
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=False)


    # ### Extract Features

    # ##### 3D ResNeXt-101


    from models import resnext

    # construct model architecture
    model_rxt101 = resnext.resnet101(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.resnet_shortcut,
                    cardinality=opt.resnext_cardinality,
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration)

    model_rxt101 = model_rxt101.cuda()
    # wrap the current model again in nn.DataParallel / or we can just remove the .module keys.
    model_rxt101 = nn.DataParallel(model_rxt101, device_ids=None)


    ### Load pretrained weight
    # customize the pretrained model path
    pretrain = torch.load(pretrain_directory_path + '/resnext-101-kinetics.pth')
    pretrain_dict = pretrain['state_dict']

    # do not load the last layer since we want to fine-tune it
    pretrain_dict.pop('module.fc.weight')
    pretrain_dict.pop('module.fc.bias')
    model_dict = model_rxt101.state_dict()
    model_dict.update(pretrain_dict) 
    model_rxt101.load_state_dict(model_dict)




    # register layer index to extract the features by forwarding all the video clips
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model_rxt101.module.avgpool.register_forward_hook(get_activation('avgpool'))
    model_rxt101.eval()


    # forward all the videos to extract features
    avgpool_test = []
    targets_test = []
    with torch.no_grad():
        print("Extract test set features:")
        for i, (inputs, target) in enumerate(test_loader):
            if i % 30 == 0:
                print(i)
            output = model_rxt101(inputs)
            avgpool_test.append(activation['avgpool'].view(len(target), -1).cpu())
            targets_test.append(target)



    avgpool_test_np = np.concatenate([i.numpy() for i in avgpool_test], axis=0)
    np.save(opt.root_path + '/extracted_features/resnext101_avgpool_test.npy', avgpool_test_np)

    targets_test_np = np.concatenate(np.array(targets_test), axis=0)
    np.save(opt.root_path + '/extracted_features/class_names_test.npy', targets_test_np)


    # ##### 3D ResNet-50


    from models import resnet

    # construct model architecture
    model_rt50 = resnet.resnet50(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.resnet_shortcut,
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration)

    model_rt50 = model_rt50.cuda()
    # wrap the current model again in nn.DataParallel / or we can just remove the .module keys.
    model_rt50 = nn.DataParallel(model_rt50, device_ids=None)


    ### Load pretrained weight
    # customize the pretrained model path
    pretrain = torch.load(pretrain_directory_path + '/resnet-50-kinetics.pth')
    pretrain_dict = pretrain['state_dict']

    # do not load the last layer since we want to fine-tune it
    pretrain_dict.pop('module.fc.weight')
    pretrain_dict.pop('module.fc.bias')
    model_dict = model_rt50.state_dict()
    model_dict.update(pretrain_dict) 
    model_rt50.load_state_dict(model_dict)




    # register layer index to extract the features by forwarding all the video clips
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model_rt50.module.avgpool.register_forward_hook(get_activation('avgpool'))
    model_rt50.eval()


    # forward all the videos to extract features
    avgpool_test = []
    with torch.no_grad():
        print("Extract test set features:")
        for i, (inputs, target) in enumerate(test_loader):
            if i % 30 == 0:
                print(i)
            output = model_rt50(inputs)
            avgpool_test.append(activation['avgpool'].view(len(target), -1).cpu())
            
        # save the features
        avgpool_test_np = np.concatenate([i.numpy() for i in avgpool_test], axis=0)
        np.save(opt.root_path + '/extracted_features/resnet50_avgpool_test.npy', avgpool_test_np)    


    # ### Load & fuse the features


    x_test_1 = np.load(opt.root_path + '/extracted_features/resnext101_avgpool_test.npy')
    x_test_2 = np.load(opt.root_path + '/extracted_features/resnet50_avgpool_test.npy')
    x_test = np.concatenate([x_test_1, x_test_2], axis=1)

    y_test = np.load(opt.root_path + '/extracted_features/class_names_test.npy')


    # ### Load Classification head and predict

    if args.model == 'hw4':
        # hw4 best model
        clf = load('./hw6_results/logistic2_ucf.joblib') 
        y_pred_test_raw = clf.predict(x_test_2)
        y_pred_test_prob_raw = clf.predict_proba(x_test_2)

    elif args.model == 'hw5':
        # hw5 best model
        clf = load('./hw6_results/logistic_ucf.joblib') 
        y_pred_test_raw = clf.predict(x_test)
        y_pred_test_prob_raw = clf.predict_proba(x_test)

    elif args.model == 'hw6':
        # hw6 best model
        clf = load('./hw6_results/logistic1_ucf.joblib') 
        y_pred_test_raw = clf.predict(x_test_1)
        y_pred_test_prob_raw = clf.predict_proba(x_test_1)

    elif args.model == 'hw8':
        # hw8 best model
        clf = load('./hw8_results/logistic_ucf.joblib') 
        y_pred_test_raw = clf.predict(x_test)
        y_pred_test_prob_raw = clf.predict_proba(x_test)

    elif args.model == 'final':
        # Final best model
        clf = load('./hw8_results/logistic1_ucf.joblib') 
        y_pred_test_raw = clf.predict(x_test_1)
        y_pred_test_prob_raw = clf.predict_proba(x_test_1)



    split_idx = []
    for idx, y_name in enumerate(y_test):
        if idx == 0 or y_name != y_test[idx-1]:
            split_idx.append(idx)
    split_idx.append(len(y_test))
            
    y_pred_test, y_pred_test_prob, y_pred_test_final = {}, {}, {}
    for i, split in enumerate(split_idx):
        if i < len(split_idx) - 1:
            y_pred_test[y_test[split]] = y_pred_test_raw[split:split_idx[i+1]]
            y_pred_test_prob[y_test[split]] = y_pred_test_prob_raw[split:split_idx[i+1]]
            y_pred_test_final[y_test[split]] = np.argmax(np.mean(y_pred_test_prob_raw[split:split_idx[i+1]], axis=0))   


    # ### Get the length (in seconds) of each video clip


    tvns = list(y_pred_test_final.keys())
    mp4_path = video_path
    clip_duration_dict = {}

    from moviepy.editor import VideoFileClip
    i = 0
    for tvn in tvns:
        i += 1
        if i % 100 == 0:
            print(i)
        clip = VideoFileClip(os.path.join(mp4_path, tvn + ".mp4"))
        clip_duration_dict[tvn] = [clip.duration]


    # ### Generate Figures
    import matplotlib.pyplot as plt
    for tvn in clip_duration_dict:
        interval = clip_duration_dict[tvn][0]/list(y_test).count(tvn)
        x = np.arange(0, clip_duration_dict[tvn][0], interval) + interval
        y_idx = np.argmax(y_pred_test_prob[tvn], 1)
        y = y_pred_test_prob[tvn][:, 1]
        x = x[:len(y)]
        plt.plot(x, y)
        plt.ylim([-0.1, 1.1])
        plt.xlabel ('time/sec')
        plt.ylabel ('pred score for ground truth label')
        plt.title("Ground Truth Label:  " + tvn  + "\n Model Avg. Predict Score:  " + str(np.mean(y))) # str(real_prediction_dict[tvn]['score'])
        plt.savefig(opt.root_path + "/final_test_results/625007598_" + tvn, bbox_inches='tight')
        plt.close()


    # ### Generate Json
    timeTrueLabel = {}
    for tvn in clip_duration_dict:
        if tvn in y_pred_test_prob:
            interval = clip_duration_dict[tvn][0]/list(y_test).count(tvn)
            x = np.arange(0, clip_duration_dict[tvn][0], interval) + interval
            y_idx = np.argmax(y_pred_test_prob[tvn], 1)
            y = y_pred_test_prob[tvn][:, 1]
            x = x[:len(y)]  
            timeTrueLabel[tvn] = [[str(time), str(y[idx])] for idx, time in enumerate(x)]



    with open(opt.root_path + '/final_test_results/625007598_timeLabel.json', 'w') as fp:
        json.dump(timeTrueLabel, fp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="CSCE689 625007598 Qingquan Song Final Test"
    )
    parser.add_argument("--video-directory-path", type=str, default="")
    parser.add_argument("--pretrain-directory-path", type=str, default="")
    parser.add_argument("--model", type=str, default="final", choices=["hw4", "hw5",  "hw6",  "hw8",  "final"])

    args = parser.parse_args()
    main(args)


