import torch
import torch.nn as nn
from argparse import ArgumentParser
import pytorch_lightning as pl
from models.unet import UNet
import torch.optim as optim
from customs.optimizers import Lookahead, RAdam
from customs.losses import get_loss
from collections import OrderedDict
from histopathology_dataset import HistopathologyDataset
from customs.mytransforms_albumentation import augmentors
from torch.utils.data import DataLoader
from customs.metrics import iou_pytorch
from customs.postprocessing import adapted_border_postprocessing
from pathlib import Path
from utils import save_image
import numpy as np


class SegModule(pl.LightningModule):
    def __init__(self, hparams):
        super(SegModule, self).__init__()
        self.hparams = hparams

        self.net = UNet(self.hparams)
        self.loss = get_loss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Parameters you define here will be available to your model through self.hparams
        """
        parser = ArgumentParser(parents=[parent_parser])
        # network params
        parser.add_argument('--ch_in', default=3, type=int, help="number of input chanels")
        parser.add_argument('--pool_method', default="conv", type=str, help="Pooling method")
        parser.add_argument('--activation_func', default="relu", type=str, help="Activation function")
        parser.add_argument('--normalization_func', default="bn", help="Normalization function")
        parser.add_argument('--filters_first_conv', default=64, type=int, help="Number of filters in first convolutional block")
        parser.add_argument('--filters_last_conv', default=1024, type=int, help="Number of filters in last convolutional block")

        parser.add_argument('--learning_rate', default=8e-4, type=float)
        parser.add_argument('--learning_rate_patience', default=9, type=int)

        # training params (opt)
        parser.add_argument('--optimizer', default='adam', type=str)
        parser.add_argument('--transforms', default='combo', type=str, help="Training data transformations.")
        parser.add_argument('--batch_size', default=6, type=int)
        parser.add_argument('--num_workers', default=12, type=int)

        parser.add_argument('--dataset_iter', default=1, type=int, help="How often should each image be in training dataset.")
        return parser

    # forward pass through the network
    def forward(self, x):
        return self.net.forward(x)

    def configure_optimizers(self):
        # Optimizer
        if self.hparams.optimizer == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                   weight_decay=0, amsgrad=True)
        elif self.hparams.optimizer == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=self.hparams.learning_rate, momentum=0.9, weight_decay=0,
                                  nesterov=True)
        elif self.hparams.optimizer == 'lookahead':
            base_optimizer = RAdam(self.net.parameters(), lr=self.hparams.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                   weight_decay=0)
            optimizer = Lookahead(base_optimizer=base_optimizer, k=5, alpha=0.5)
        else:
            raise Exception(f'Unknown optimizer: {self.hparams.optimizer}')

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25,
                                                         patience=self.hparams.learning_rate_patience, verbose=True,
                                                         min_lr=6e-5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        pred_border, pred_cell = self.forward(batch["image"])
        y_true_border = batch["label_seed"]+batch["label_border"]*2
        y_true_border = torch.unsqueeze(y_true_border, 1).to(torch.int64)
        loss_border = self.loss["border"](pred_border, y_true_border)
        y_true_cell = (torch.unsqueeze(batch["label_cell"],1)/255).to(torch.float)
        loss_cell = self.loss["cell"](pred_cell, y_true_cell)
        loss = loss_border + loss_cell
        tqdm_dict = {'train_loss': loss, 'train_loss_cell': loss_cell, 'train_loss_border': loss_border}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'progress_bar': {'training_loss': loss},
        })
        return output

    def validation_step(self, batch, batch_idx):
        pred_border, pred_cell = self.forward(batch["image"])
        if self.current_epoch % 10 == 0:
            if batch_idx == 0:
                # stack = torch.stack([batch['label_border_seed'][0,:],
                #                      batch['label_cell'][0,:],
                #                      pred_border[0,:],
                #                      pred_cell[0,:]], dim=0)
                # grid = torchvision.utils.make_grid(stack, nrow=2)
                label_border_seed = (batch['label_border'][0,:]*255).type(torch.uint8)+(batch['label_seed'][0,:]*128).type(torch.uint8)
                img = batch["image"][0,:].type(torch.uint8)
                pred_border_seed = (torch.sigmoid(pred_border[0,1,:])*128).type(torch.uint8)+(torch.sigmoid(pred_border[0,2,:])*255).type(torch.uint8)
                self.logger.experiment.add_image('Image', img, self.current_epoch, dataformats='CHW')
                self.logger.experiment.add_image('Label/cell', batch['label_cell'][0,:], self.current_epoch, dataformats='HW')
                self.logger.experiment.add_image('Prediction/cell', (torch.sigmoid(pred_cell[0,0,:])*255).type(torch.uint8), self.current_epoch, dataformats='HW')
                self.logger.experiment.add_image('Label/border_seed', label_border_seed, self.current_epoch, dataformats='HW')
                self.logger.experiment.add_image('Prediction/border_seed', pred_border_seed, self.current_epoch, dataformats='HW')
                self.logger.experiment.add_image('Prediction/seed', (torch.sigmoid(pred_border[0,1,:])*255).type(torch.uint8), self.current_epoch, dataformats='HW')
                self.logger.experiment.add_image('Prediction/border', (torch.sigmoid(pred_border[0,2,:])*255).type(torch.uint8), self.current_epoch, dataformats='HW')
        y_true = batch["label_seed"]+batch["label_border"]*2
        y_true = torch.unsqueeze(y_true, 1).to(torch.int64)
        loss_border = self.loss["border"](pred_border, y_true)
        y_true = (torch.unsqueeze(batch["label_cell"],1)/255).to(torch.float)
        loss_cell = self.loss["cell"](pred_cell, y_true)
        val_loss = loss_border + loss_cell


        # IOU - Metric
        iou_border = iou_pytorch(pred_border[:, 1, :, :], batch["label_seed"])
        iou_cells = iou_pytorch(pred_cell, batch["label_cell"])
        iou = (iou_border + iou_cells) / 2

        output = OrderedDict({
            'val_loss_t': val_loss,
            'val_loss_border_t': loss_border,
            'val_loss_cell_t': loss_cell,
            'val_iou_t': iou
        })
        return output

    def validation_epoch_end(self, output):
        loss_border = torch.stack([x['val_loss_border_t'] for x in output]).mean()
        loss_cell = torch.stack([x['val_loss_cell_t'] for x in output]).mean()
        iou = torch.stack([x['val_iou_t'] for x in output]).mean()
        val_loss = loss_border+loss_cell
        tqdm_dict = {'val_loss': val_loss, 'val_loss_border': loss_border, 'val_loss_cell': loss_cell, 'vall_IoU': iou, 'step': self.current_epoch}
        output = OrderedDict({
            'log': tqdm_dict
        })
        return output

    def test_step(self, batch, batch_idx):
        pred_border, pred_cell = self.forward(batch["image"])
        pred_border = torch.nn.functional.softmax(pred_border, dim=1)
        pred_cell = torch.sigmoid(pred_cell)
        for i in range(0, pred_cell.shape[0]):
            border = pred_border[i, :, :, :].permute(1, 2, 0).cpu().numpy()
            cell = pred_cell[i, 0, :, :].cpu().numpy()
            folder = Path(self.hparams.root_dir).joinpath("results", self.hparams.experiment)
            if self.hparams.save_raw_pred:
                cell_path = folder.joinpath(batch["id"][i]+"_raw_cell.npy")
                border_path = folder.joinpath(batch["id"][i]+"_raw_border.npy")
                raw_path = folder.joinpath(batch["id"][i]+"_raw.npz")
                raw_path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(raw_path, border=border, cell=cell)
            prediction_instance, prediction_instance_rgb, pred_border_bin = adapted_border_postprocessing(border, cell)
            instance_path = folder.joinpath(batch["id"][i]+"_instance.png")
            instance_rgb_path = folder.joinpath(batch["id"][i]+"_instance_rgb.png")
            border_path = folder.joinpath(batch["id"][i]+"_border.png")
            save_image(prediction_instance, instance_path)
            save_image(prediction_instance_rgb, instance_rgb_path)
            save_image(pred_border_bin, border_path)
        print(f"Batch {batch_idx} processed in test_step")
        #return {'blank': 0}

    def test_epoch_end(self, outputs):
        print("Inference finished!")
        return {'nologs':0}

    def train_dataloader(self):
        transforms = augmentors(self.hparams.transforms, label_type='adapted_border', min_value=0, max_value=255)
        self.train_dataset = HistopathologyDataset(self.hparams.root_dir, mode='train', transforms=transforms["train"], dataset_iter=self.hparams.dataset_iter)
        loader = DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)
        return loader

    def val_dataloader(self):
        transforms = augmentors(self.hparams.transforms, label_type='adapted_border', min_value=0, max_value=255)
        self.val_dataset = HistopathologyDataset(self.hparams.root_dir, mode='val', transforms=transforms["val"])
        loader = DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
        return loader

    def test_dataloader(self):
        transforms = augmentors(self.hparams.transforms, label_type=None, min_value=0, max_value=255)
        self.val_dataset = HistopathologyDataset(self.hparams.root_dir, mode='test', transforms=transforms["test"])
        loader = DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
        return loader
