import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from pathlib import Path
import numpy as np
import shutil
import hjson


def find_means():
    trans = transforms.ToTensor()
    dataset = datasets.ImageFolder("/srv/user/boehland/Luebeck/Lars/Data/Original/split/classy_vision/train", transform=trans)
    means = torch.zeros(len(dataset),3)
    stds = torch.zeros(len(dataset),3)
    for idx, img in enumerate(dataset):
        means[idx,:] = torch.tensor([img[0][0,:].mean(), img[0][1,:].mean(), img[0][2,:].mean()])
        stds[idx,:] = torch.tensor([img[0][0,:].std(), img[0][1,:].std(), img[0][2,:].std()])
    mean = means.mean(0)
    std = stds.mean(0)
    return mean, std


def accuracy_topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_paths(folder, suffix):
    image_paths = []
    for file in folder.iterdir():
        if file.suffix == suffix:
            image_paths.append(file)
    return np.asarray(image_paths)
