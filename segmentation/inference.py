import pytorch_lightning as pl
from argparse import ArgumentParser
from seg_module import SegModule
from pytorch_lightning.logging import TestTubeLogger
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

"""Use to predict the instance segmentation on new data
Set --root_dir to the direction with the images to predict
Set --checkpoint_path to the checkpoint file of the trained model. 
Checkpoints are stored in the folder results/EXPERIMENTNAME_VERSION/*.ckpt
A greyscale instance segmentation (_instance) a rgb instance segmentation (_instance_rgb)
and the border prediction (_border) will be saved to a results folder in the root_dir


If images are not present in .tif please change the ending in the init of histopathology_dataset.py file accordingly.

The results will be saved into results/EXPERIMENT_NAME folder created in the rood_dir_default folder.
EXPERIMENT_NAME is the name the experiment had during training.
For the Tharun and Tompson dataset and the Nikiforov dataset change the EXPERIMENT_NAME to 'segmentation' afterwards
"""



def main(hparams):
    trainer = pl.Trainer(gpus=1)
    model = SegModule.load_from_checkpoint(checkpoint_path)
    model.hparams.num_workers = 0  # Change number of workers to 0 to be able to debug
    model.hparams.checkpoint_path = hparams.checkpoint_path
    model.hparams.root_dir = hparams.root_dir
    model.hparams.batch_size = 1  # To be able to handle different image sizes
    model.hparams.save_raw_pred = hparams.save_raw_pred
    trainer.test(model)


if __name__ == '__main__':
    root_dir_default = Path(__file__).parent.joinpath("../datasets/MoNuSeg/val/Tissue Images").resolve()
    checkpoint_path = Path(__file__).parent.joinpath("../datasets/MoNuSeg/epoch=524-val_loss=1.21.ckpt").resolve()
  
    
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("--root_dir", type=str, default=root_dir_default, help="path where dataset is stored")
    parent_parser.add_argument("--experiment", default="histo_test", type=str, help="Name of the experiment.")
    parent_parser.add_argument("--checkpoint_path", default=checkpoint_path, type=str, help="Path to the saved checkpoint")
    parent_parser.add_argument("--save_raw_pred", default=False, help="Save the raw prediction (cell, border/seed) during inference or not.")
    parser = SegModule.add_model_specific_args(parent_parser, root_dir_default)
    hparams = parser.parse_args()
    main(hparams)
