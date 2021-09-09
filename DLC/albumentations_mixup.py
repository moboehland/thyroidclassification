from albumentations.core.transforms_interface import BasicTransform, to_tuple
from albumentations.augmentations.utils import read_rgb_image
import random
import cv2
import torch
import torch.nn as nn


class Mixup(BasicTransform):
    def __init__(self, mixups, read_fn=read_rgb_image, beta_limit=0.3, mixup_normalization=None, **kwargs):
        super().__init__(**kwargs)
        self.mixups = mixups
        self.read_fn = read_fn
        self.beta_limit = to_tuple(beta_limit, low=0)
        self.mixup_normalization = mixup_normalization

    def apply(self, image, mixup_image=None, beta=0.1, **params):
        img_type = image.dtype
        mixup_image = self.mixup_normalization(image=mixup_image)["image"]
        image = ((1-beta)*image+beta*mixup_image).astype(img_type)
        return image

    def apply_to_target(self, target, beta=0.1, mixup_target=-1, **params):
        target = {"img": target, "mixup": mixup_target, "beta": beta}
        return target

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        mixup = random.choice(self.mixups)
        mixup_image = self.read_fn(mixup[0])
        mixup_image = cv2.resize(mixup_image, dsize=(img.shape[1], img.shape[0]))
        return {"mixup_image": mixup_image, "mixup_target": mixup[1]}

    def get_params(self):
        return {"beta": random.uniform(self.beta_limit[0],self.beta_limit[1])}

    @property
    def targets(self):
        return {
            "image": self.apply,
            "target": self.apply_to_target,
        }

    @property
    def targets_as_params(self):
        return ["image"]


def mixup_loss(output, target):
    if type(target) == torch.Tensor:
        loss = nn.CrossEntropyLoss()
        return loss(output, target)
    else:  # mixup has been used
        loss = nn.CrossEntropyLoss(reduction="none")
        return ((1-target["beta"])*loss(output, target["img"])+target["beta"]*loss(output, target["mixup"])).mean()
