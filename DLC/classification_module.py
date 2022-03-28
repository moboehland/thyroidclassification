import torch
import torch.nn as nn
import pytorch_lightning as pl
from argparse import ArgumentParser
from classification_nets import get_classification_net
from utils import accuracy_topk
from thyroid_dataset import ThyroidDataset as ClassificationDataset
from pathlib import Path
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations_mixup import Mixup
import random


class ClassificationModule(pl.LightningModule):

    def __init__(self,
                 net="resnet50",
                 classes=2,
                 pretrained=True,
                 batch_size=32,
                 num_workers=0,
                 dataset_folder="",
                 learning_rate=0.001,
                 lr_patience=25,
                 lr_reduce_factor=0.5,
                 use_lr_finder=False,
                 augmentation="min",
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.net = get_classification_net(self.hparams.net, self.hparams.pretrained, self.hparams.classes)
        self.loss =  nn.CrossEntropyLoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # network params
        parser.add_argument('--net', default="resnet50", type=str, help="Net architecture to use")
        parser.add_argument('--classes', default=2, type=int, help="Number of output classes")
        parser.add_argument('--pretrained', default=False, action="store_true", help="Enable using pretrained models.")
        # dataloader params
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--num_workers', default=16, type=int)
        # dataset params
        parser.add_argument('--class_imbalance_threshold', default=100, type=int, help="Maximal allowed class imbalance in dataset in percent. If <100 samples in small classes are picked multiple times for dataset. Only applied for trainind data (not val/test).")
        parser.add_argument('--sliding_window', default=0, type=int, help="size of sliding window (rectangle) to use for the dataset.")
        # optimizer params
        parser.add_argument('--learning_rate', default=0.001, type=float, help="Learning rate of optimizer. Will only be used if use_lr_finder is not set.")
        parser.add_argument('--use_lr_finder', default=False, action="store_true", help="Enable learning rate finder. Argument learning_rate will be used otherwise.")
        # scheduler params
        parser.add_argument('--scheduler', default="ReduceLROnPlateau", type=str, help="used scheduler [ReduceLROnPlateau, CosineAnnealingWarmRestarts]")
        parser.add_argument('--lr_patience', default=25, type=int, help="ReduceLROnPlateau: Patience for learning rate reducement")
        parser.add_argument('--lr_reduce_factor', default=0.5, type=float, help="ReduceLROnPlateau: Factor by which the learning rate will be reduced.")
        parser.add_argument('--weight_decay', default=0, type=float, help="weight decay (L2 penalty)")
        parser.add_argument('--T_0', default=25, type=int, help="CosineAnnealingWarmRestarts: Number of iterations for the first restart")
        parser.add_argument('--T_mult', default=1, type=int, help="CosineAnnealingWarmRestarts: Factor increasing T_i after restart")

        # Albumentations
        parser.add_argument('--augmentation', default="min", type=str, help="Use Augmentation. min (minimal), std (standard) or path to the auto-Albumentations json file.")
        parser.add_argument('--crop_train', default=0, type=int, help="Crop image to X by X. If X is 0 no cropping will be performed")
        parser.add_argument('--crop_val', default=0, type=int, help="Crop image to X by X. If X is 0 no cropping will be performed")
        parser.add_argument('--crop_test', default=0, type=int, help="Crop image to X by X. If X is 0 no cropping will be performed")
        return parser

    def forward(self, x):
        return self.net.forward(x)

    def training_step(self, batch, batch_idx):  # batch = {"image": sample["image"], "target": sample["target"], "path": path, "window": window}
        pred = self(batch["image"])
        loss = self.loss(pred, batch["target"])
        train_top1 = accuracy_topk(pred, batch["target"], topk=(1,)) #Only topk=(X,) supported atm by logging (otherwise train_topk is tuple which can not be logged to tensorboard easily)
        self.log('train/loss', loss, on_epoch=True, on_step=False)
        self.log('train/top1', train_top1[0][0], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["image"])
        loss = self.loss(pred, batch["target"])
        val_top1 = accuracy_topk(pred, batch["target"], topk=(1,))  # Only topk=(X,) supported atm by logging (otherwise train_topk is tuple which can not be logged to tensorboard easily)
        self.log('val/loss', loss, on_epoch=True, logger=True, on_step=False)
        self.log('val/top1', val_top1[0][0], on_epoch=True, logger=True, on_step=False)
        return

    def test_step(self, batch, batch_idx):
        pred = self(batch["image"])
        pred_vals, pred_ind = pred.topk(1)
        for idx, pred_i in enumerate(pred_ind):
            if pred_i.item() != batch["target"][idx]:
                print(f"Pred: {pred_i.item()}, GT: {batch['target'][idx]}, Img:{Path(batch['path'][idx]).name}")
        val_top1 = accuracy_topk(pred, batch["target"], topk=(1,))  # Only topk=(X,) supported atm by logging (otherwise train_topk is tuple which can not be logged to tensorboard easily)
        
        return {'val_top1': val_top1[0][0], "pred_ind": pred_ind.squeeze(dim=1), "gt_ind": batch["target"], "gt_paths": batch["path"], "window": batch["window"]}

    def test_epoch_end(self, outputs):
        class_idx = self.trainer.test_dataloaders[0].dataset.class_to_idx
        class_idx = dict(zip(class_idx.values(), class_idx.keys())) # change key and value
        val_top1 = torch.stack([x['val_top1'] for x in outputs]).mean()
        results = pd.DataFrame(columns=["gt_paths", "gt_ind", "pred_ind", "gt_name", "pred_name", "window"])
        for idx, row in enumerate(outputs):
            gt_paths = list(row["gt_paths"])
            gt_ind = row["gt_ind"].tolist()
            pred_ind = row["pred_ind"].tolist()
            window = row["window"].tolist()
            pred_name = [class_idx[ind] for idx, ind in np.ndenumerate(pred_ind)]  # np.ndenumerate because last item can be int and no list
            gt_name = [class_idx[ind] for idx, ind in np.ndenumerate(gt_ind)]
            pd_row = pd.DataFrame(list(zip(gt_paths, gt_ind, pred_ind, gt_name, pred_name, window)), columns=["gt_paths", "gt_ind", "pred_ind", "gt_name", "pred_name", "window"])
            results = results.append(pd_row, ignore_index=True)
        print(f"Val top1: {val_top1}")
        self.results = results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler == "ReduceLROnPlateau":
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=self.hparams.lr_patience, verbose=True, factor=self.hparams.lr_reduce_factor),
                'monitor': 'val/loss'
            }
        elif self.hparams.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.T_0, T_mult=self.hparams.T_mult, verbose=True),
            }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # TharunThompson Dataset:
        # Training-means: 0.6541, 0.4286, 0.6997
        # Training-stds:  0.1614, 0.1608, 0.1187
        # pytorch mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225
        if ".json" in self.hparams.augmentation:
            trans = A.load(self.hparams.augmentation)
        elif self.hparams.augmentation == "min":
            trans = A.Compose([
                (lambda x: A.RandomCrop(x, x, p=1) if x > 0 else A.RandomCrop(x, x, p=0))(self.hparams.crop_train),
                A.Flip(),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        elif self.hparams.augmentation == "std":
            trans = A.Compose([
                A.Flip(p=0.5),
                A.Rotate(p=0.5),
                (lambda x: A.RandomCrop(x, x, p=1) if x > 0 else A.RandomCrop(x, x, p=0))(self.hparams.crop_train),
                A.OneOf([A.CLAHE(p=0.33),
                         A.RandomContrast(p=0.33),
                         A.RandomBrightnessContrast(p=0.33)],
                        p=0.5),
                A.Blur(p=0.25),
                A.GaussNoise(p=0.25),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        elif self.hparams.augmentation == "fda":
            dataset = ClassificationDataset(self.hparams.dataset_folder, self.hparams.split["train"],
                                            class_imbalance_threshold=self.hparams.class_imbalance_threshold)
            fda_image_paths = [sample[0] for sample in dataset.samples]
            trans = A.Compose([
                A.Flip(p=0.5),
                A.Rotate(p=0.5),
                (lambda x: A.RandomCrop(x, x, p=1) if x > 0 else A.RandomCrop(x, x, p=0))(self.hparams.crop_train),
                A.domain_adaptation.FDA(fda_image_paths, beta_limit=0.05, p=0.5),
                A.OneOf([A.CLAHE(p=0.33),
                         A.RandomContrast(p=0.33),
                         A.RandomBrightnessContrast(p=0.33)],
                        p=0.5),
                A.Blur(p=0.25),
                A.GaussNoise(p=0.25),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        elif self.hparams.augmentation == "mixup":
            dataset = ClassificationDataset(self.hparams.dataset_folder, self.hparams.split["train"],
                                            class_imbalance_threshold=self.hparams.class_imbalance_threshold)
            mixups = [sample[0:2] for sample in dataset.samples]
            trans = A.Compose([
                A.Flip(p=0.5),
                A.Rotate(p=0.5),
                (lambda x: A.RandomCrop(x, x, p=1) if x > 0 else A.RandomCrop(x, x, p=0))(self.hparams.crop_train),
                Mixup(mixups=mixups, p=1, beta_limit=(0.5),
                      mixup_normalization=A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))),
                A.OneOf([A.CLAHE(p=0.33),
                         A.RandomContrast(p=0.33),
                         A.RandomBrightnessContrast(p=0.33)],
                        p=0.5),
                A.Blur(p=0.25),
                A.GaussNoise(p=0.25),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])


        else:
            raise ValueError("Augmentation unknown")

        dataset = ClassificationDataset(self.hparams.dataset_folder, self.hparams.split["train"], transform=trans,
                                        class_imbalance_threshold=self.hparams.class_imbalance_threshold)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, worker_init_fn=self.seed_worker)  #, worker_init_fn=self.seed_worker
        return train_loader

    def val_dataloader(self):
        trans = A.Compose([
            (lambda x: A.RandomCrop(x, x, p=1) if x > 0 else A.RandomCrop(x, x, p=0))(self.hparams.crop_val),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        dataset = ClassificationDataset(self.hparams.dataset_folder, self.hparams.split["val"], transform=trans,
                                        sliding_window=self.hparams.sliding_window)
        if (self.hparams.crop_val <= 0) & (self.hparams.sliding_window <= 0):
            # images are not cropped and no sliding_window is applied batch size has to be one since not all images have the same size
            batch_size = 1
        else:
            batch_size = self.hparams.batch_size
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.hparams.num_workers)  # batch-size 1 (no crop => different image sizes)
        return val_loader

    def test_dataloader(self):
        trans = A.Compose([
            (lambda x: A.RandomCrop(x, x, p=1) if x > 0 else A.RandomCrop(x, x, p=0))(self.hparams.crop_test),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        dataset = ClassificationDataset(self.hparams.dataset_folder, self.hparams.split["test"], transform=trans, sliding_window=self.hparams.sliding_window)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.hparams.num_workers)
        return test_loader

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
