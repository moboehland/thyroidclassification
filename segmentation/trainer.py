import pytorch_lightning as pl
from argparse import ArgumentParser
from seg_module import SegModule
from pytorch_lightning.logging import TestTubeLogger
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

""" Training script for the segmentation network on the MoNuSeg Challenge data
See README.md for instructions

The final model has been trained with the following hyperparameters different from the standard hyperparameters
--filters_last_conv 2048

Our final model has a val_loss of 1.21. It is possible that you need to train several networks with the same hyperparameter settings to achieve this.
Trained models are saved to results/EXPERIMENT_VERSION/*.ckpt. Tensorboard logs during training are saved to lightning_logs/EXPERIMENT/VERSION.

To train the model on the MoNuSeg Challenge data, the directory needs to be given using the argument parser (e.g.: --root_dir ../datasets/MoNuSeg)
"""

def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SegModule(hparams)


    # DEFAULTS used by the Trainer
    i = 1
    exp_name = hparams.experiment
    hparams.experiment = exp_name+"_V"+str(i)
    while Path(__file__).resolve().parent.joinpath("results", hparams.experiment).exists():
        hparams.experiment = exp_name+"_V"+str(i)
        i = i+1

    checkpoint_callback = ModelCheckpoint(
        filepath=Path(__file__).resolve().parent.joinpath("results", hparams.experiment, "{epoch}-{val_loss:.2f}"),
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    # ------------------------
    # CREATE LOGGER
    # ------------------------
    tt_logger = TestTubeLogger(save_dir=Path(__file__).resolve().parent.joinpath("results"),
                               name=hparams.experiment,
                               debug=False)


    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=50,
    )

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    tt_logger.log_hyperparams(hparams)
    trainer = pl.Trainer(
        gpus=1,
        logger=tt_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=1000,
        min_epochs=100,
        benchmark=True,
        check_val_every_n_epoch=1,                  # needs to be one because of scheduler
        early_stop_callback=early_stop_callback
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':  
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("--root_dir", type=str, default="datasets/MoNuSeg", help="path where dataset is stored")
    parent_parser.add_argument("--experiment", default="test_experiment", type=str, help="Name of the experiment.")
    parent_parser.add_argument("--early_stop", default=25, type=int, help="Epochs without better val_loss before ending training.")
    parser = SegModule.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()
    main(hparams)
