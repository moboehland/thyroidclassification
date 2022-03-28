import albumentations as albu
from albumentations.pytorch import ToTensorV2
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))


def augmentors(augmentation, min_value=0, max_value=255, label_type=None):
    if augmentation == "combo":
        targets = {'label_border': 'mask', 'label_seed': 'mask', 'label_cell': 'mask'}
        transforms_train = [albu.Flip(p=0.75),
                            albu.RandomScale(scale_limit=0.2, p=0.3),
                            albu.Rotate(limit=180, p=0.3),
                            albu.OneOf([albu.CLAHE(p=0.5),
                                        albu.RandomBrightnessContrast(p=0.5),
                                        albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1)], p=0.5),  #0.3
                            albu.OneOf([albu.Perspective(p=0.5),
                                        albu.PiecewiseAffine(p=0.5, interpolation=0)], p=0.5),  #0.25
                            albu.RandomCrop(384, 384, p=1),
                            albu.Blur(p=0.25),  # 0.25
                            albu.GaussNoise(p=0.25),  # 0.25
                            ToTensorV2()]
        transforms_train = albu.Compose(transforms_train, p=1, additional_targets=targets)
        transforms_val = albu.Compose([albu.RandomCrop(992, 992, p=1), ToTensorV2()], p=1, additional_targets=targets) #, albu.Normalize(p=1.0)
        transforms_test = albu.Compose([ToTensorV2()], p=1)
        trans = {'train': transforms_train, 'val': transforms_val, 'test': transforms_test}
    else:
        raise Exception('Unknown transformation: {}'.format(augmentation))
    return trans
