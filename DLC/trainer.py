import pytorch_lightning as pl
from argparse import ArgumentParser
from classification_module import ClassificationModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import json
from pathlib import Path
from pytorch_lightning.loggers import TestTubeLogger
import pandas as pd
import numpy as np
import sys


def load_splits(splits_folder):
    split_paths = list(splits_folder.glob("*.json"))
    split_paths.sort()
    splits = []
    for split_path in split_paths:
        with open(split_path) as infile:
            splits.append(json.load(infile))
    if splits == []:
        raise (ValueError(f"No splits found in folder {splits_folder}"))
    return splits, split_paths


def splitfilenames_from_splits(data_path, splits):
    image_paths = list(Path(data_path).glob("*.*"))  # all files no directories
    ground_truth = pd.read_excel(data_path.joinpath("ground_truth.xlsx"), engine="openpyxl")
    splits_filenames = []
    for split in splits:
        splits_filenames.append(dict.fromkeys(splits[0].keys()))
        for key in split:
            image_filenames = []
            for patient in split[key]:
                for image_path in image_paths:
                    # if patient is integer, use old format and compare integers (patient=100, all images 100a, 100b, ...)
                    try:
                        if int(patient) == int(image_path.stem[:-1]):
                            image_filenames.append((image_path.name, ground_truth[ground_truth["sample"]==patient]["diagnose_grouped"].iloc[0]))
                    except ValueError:  # filename is explicitely given (e.g. patient=100a, only image 100a is added)
                        if patient == image_path.stem:
                            image_filenames.append((image_path.name, ground_truth[ground_truth["sample"]==patient]["diagnose_grouped"].iloc[0]))
            splits_filenames[-1][key] = image_filenames
    return splits_filenames


def train_model(hparams, split_filenames):
    hparams.split = split_filenames

    # change number of saved checkpoints
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,  # Display decisions in console (Save new checkpoint or not)
        monitor='val/loss',
        mode='min',  # topk needs to be maximized
        prefix=''
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        mode='min',  # topk needs to be maximized
        patience=hparams.early_stop_patience,  # 30
    )

    
    logger = TestTubeLogger(str(Path(__file__).parent.joinpath("lightning_logs")), name=hparams.dataset_folder.name+"/"+hparams.name+"/"+hparams.split_filename)
    

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, auto_lr_find=hparams.use_lr_finder, callbacks=[checkpoint_callback, early_stop_callback])
    model = ClassificationModule(**vars(args))
    trainer.tune(model)  # findes auto_lr if set
    trainer.fit(model)


def test_model(hparams, checkpoint_path):
    model = ClassificationModule.load_from_checkpoint(checkpoint_path=str(checkpoint_path))
    model.hparams["sliding_window"] = hparams.sliding_window
    model.hparams["dataset_folder"] = hparams.dataset_folder
    trainer = pl.Trainer.from_argparse_args(args, logger=None, progress_bar_refresh_rate=0)
    trainer.test(model)

    return model.results


def results_image_to_patient(results_image):
    results_image["patient"] = [int(Path(p).stem[:-1]) for p in results_image["gt_paths"]]
    results_classes = results_image["gt_ind"].unique()
    results_patient = pd.DataFrame()
    for patient in results_image["patient"].unique():
        res_p = results_image[results_image["patient"]==patient]
        n_classifications = np.array([len(res_p[res_p["pred_ind"]==results_class]) for results_class in results_classes])
        if np.count_nonzero(n_classifications==n_classifications.max())==1:
            pred_ind = results_classes[n_classifications.argmax()]
        else:
            pred_ind = -1  # multiple classes have the same maximum count therefore no clear prediction can be given
        result_patient = pd.DataFrame([res_p.iloc[0][["patient", "gt_ind", "gt_name", "split", "rep"]]], columns=["patient", "gt_ind", "gt_name", "split", "rep"])
        result_patient = pd.concat([result_patient.reset_index(), pd.DataFrame([[pred_ind, *n_classifications]], columns=["pred_ind", *results_classes])], axis=1, sort=False)
        results_patient = results_patient.append(result_patient, ignore_index=True)

    return results_patient


def get_val_loss_top1(metrics_folder):
    metrics = pd.read_csv(metrics_folder.joinpath("metrics.csv"), delimiter=",")
    val_loss = metrics["val/loss"].min()
    val_top1 = metrics.iloc[metrics["val/loss"].argmin()]["val/top1"]
    return val_loss, val_top1


def main(hparams):
    splits, split_paths = load_splits(hparams.splits_folder)
    splits_filenames = splitfilenames_from_splits(hparams.dataset_folder, splits)
    if hparams.mode == "train":
        for idx, split_filenames in enumerate(splits_filenames):
            for rep in range(hparams.n_reps):
                hparams.rep = rep
                hparams.split_filename = split_paths[idx].stem
                train_model(hparams, split_filenames)
    elif hparams.mode == "test":
        results_image = pd.DataFrame()
        results_patient = pd.DataFrame()
        results_split = pd.DataFrame()
        results_folder = Path(__file__).parent.joinpath("lightning_logs", hparams.dataset_folder.name, hparams.name).absolute()
        for split in split_paths:
            split_name = split.stem
            split_folder = results_folder.joinpath(split_name)
            rep_paths = list(split_folder.glob("*"))
            rep_paths.sort()
            for rep in rep_paths:
                checkpoint_folder = rep.joinpath("checkpoints")
                checkpoint_filename = list(checkpoint_folder.glob("*"))[0].name
                checkpoint_path = checkpoint_folder.joinpath(checkpoint_filename)
                results_image_rep = test_model(hparams, checkpoint_path)
                results_image_rep["split"] = split.stem
                results_image_rep["rep"] = rep.stem
                results_image = results_image.append(results_image_rep, ignore_index=True)
                results_patient_rep = results_image_to_patient(results_image_rep)
                results_patient_rep["right_pred"] = results_patient_rep["gt_ind"]==results_patient_rep["pred_ind"]
                results_patient = results_patient.append(results_patient_rep, ignore_index=True)

                val_loss, val_top1 = get_val_loss_top1(rep)
                test_top1 = results_patient_rep["right_pred"].sum() / len(results_patient_rep["right_pred"]) * 100
                results_split = results_split.append(pd.DataFrame([[split.stem, rep.stem, val_loss, val_top1, test_top1]],
                                                                  columns=["split", "rep", "val/loss", "val/top1", "test/top1"]))
                
        # calculate final top1 score from results_split (pick best rep (by val/loss) for each split)
        result_final = pd.DataFrame()
        results_split.reset_index(inplace=True, drop=True)   # to use idxmax()
        for split in results_split["split"].unique():
            val_loss_idxmin = results_split[results_split["split"]==split]["val/loss"].idxmin()
            result_final = result_final.append(results_split.loc[val_loss_idxmin], sort=False)

        # calculate final results on patient basis for best splits
        results_final_patient = pd.DataFrame()
        for idx, row in result_final.iterrows():
            split_name = row["split"]
            rep = row["rep"]
            results_final_patient = results_final_patient.append(results_patient[(results_patient["split"]==split_name) & (results_patient["rep"]==rep)])

        result_final = result_final[results_split.columns] # rearange columns
        result_final_mean = pd.DataFrame([["mean", "mean", result_final.mean()["val/loss"], result_final.mean()["val/top1"], result_final.mean()["test/top1"]]], columns=result_final.columns)
        result_final = result_final.append(result_final_mean, ignore_index=True)
        writer = pd.ExcelWriter(results_folder.joinpath("results_split.xlsx"), engine='openpyxl')
        results_split.to_excel(writer, sheet_name='split_results')
        result_final.to_excel(writer, sheet_name='final_result')
        results_final_patient.to_excel(writer, sheet_name='final_results_patient')
        writer.save()
        writer.close()
        results_patient.to_excel(results_folder.joinpath("results_patient.xlsx"))
        results_image.to_excel(results_folder.joinpath("results_image.xlsx"))


if __name__ == '__main__':
    if pl.__version__ != "1.0.8":
        raise (RuntimeError(f"Pytorch Lightning version {pl.__version__} found, version 1.0.8 is required! If you have installed a higher version, remove this code at own risk."))
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--benchmark', default=True, action="store_false", help="Disable if input size changes. Benchmark can increases training speed for same sized inputs.")
    parser.add_argument("--dataset_folder", type=str, default="../datasets/TharunThompson/", help="Define folder to the dataset. Can be relative path from this script.")
    parser.add_argument("--splits_folder", type=str, default="../datasets/TharunThompson/results/FBC/test/splits", help="Define folder for the splits created during the feature based classification.  Can be relative path from this script.")
    parser.add_argument("--name", type=str, default="test", help="Name of the experiment.")
    parser.add_argument("--n_reps", type=int, default=3, help="Number of repetitions the network is trained for e ach split")
    parser.add_argument("--mode", type=str, default="train", help="train or test mode.")
    parser.add_argument("--early_stop_patience", type=int, default=50, help="Patience of the early stopping callback.")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="If you have a small dataset you might want to check validation every n epochs")
    parser = ClassificationModule.add_model_specific_args(parser)
    args = parser.parse_args()
    
    args.dataset_folder = Path(__file__).parent.joinpath(args.dataset_folder)
    args.splits_folder = Path(__file__).parent.joinpath(args.splits_folder)
    main(args)
