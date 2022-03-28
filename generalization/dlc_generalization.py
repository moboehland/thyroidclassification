from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re
import sys
sys.path.append(str(Path(__file__).parents[1].joinpath("DLC")))
from DLC import ClassificationModule

""" Script to evaluate the performance of the best feature based classification pipeline trained on one dataset evaluated on another dataset
The direction (TtoN or NtoT) and the paths to the corresponding trained networks need to be set manually. 
"""

direction = "NtoT"  # TtoN or NtoT
networks_T = [Path(__file__).parents[0].joinpath(*["dlc_trained_networks", "TharunThompson", "TharunThompson_split0_rep2.ckpt"]),
              Path(__file__).parents[0].joinpath(*["dlc_trained_networks", "TharunThompson", "TharunThompson_split1_rep1.ckpt"]),
              Path(__file__).parents[0].joinpath(*["dlc_trained_networks", "TharunThompson", "TharunThompson_split2_rep2.ckpt"]),
              Path(__file__).parents[0].joinpath(*["dlc_trained_networks", "TharunThompson", "TharunThompson_split3_rep1.ckpt"]),
              Path(__file__).parents[0].joinpath(*["dlc_trained_networks", "TharunThompson", "TharunThompson_split4_rep4.ckpt"])]

networks_N = [Path(__file__).parents[0].joinpath(*["dlc_trained_networks", "Nikiforov", "Nikiforov_split0_rep0.ckpt"]),
              Path(__file__).parents[0].joinpath(*["dlc_trained_networks", "Nikiforov", "Nikiforov_split1_rep0.ckpt"]),
              Path(__file__).parents[0].joinpath(*["dlc_trained_networks", "Nikiforov", "Nikiforov_split2_rep2.ckpt"]),
              Path(__file__).parents[0].joinpath(*["dlc_trained_networks", "Nikiforov", "Nikiforov_split3_rep3.ckpt"]),
              Path(__file__).parents[0].joinpath(*["dlc_trained_networks", "Nikiforov", "Nikiforov_split4_rep0.ckpt"])]



N_folder_20x = Path(__file__).parent.joinpath("..", "datasets", "Nikiforov").resolve()
N_folder_40x = Path(__file__).parent.joinpath("..", "datasets", "Nikiforov_upscale2x").resolve()
T_folder_40x = Path(__file__).parent.joinpath("..", "datasets", "TharunThompson").resolve()
T_folder_20x = Path(__file__).parent.joinpath("..", "datasets", "TharunThompson_downscale2x").resolve()

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',
                  '.TIFF', '.tif', '.TIF']

def get_image_paths(folder, substring=None):
    image_paths = []
    for file in folder.iterdir():
        if any(file.suffix == extension for extension in IMG_EXTENSIONS):
            if substring==None:
                image_paths.append(file)
            else:
                if substring in file.name:
                    image_paths.append(file)
    return np.asarray(image_paths)

def merge_path_gt(image_paths, ground_truth, dataset):
    patient_numbers = np.zeros(len(image_paths))
    diagnose_grouped = []
    T_paths = np.asarray(image_paths)
    for i, image_path in enumerate(image_paths):
        # if patient has multiple images e.g. 1a, 1b, ... a,b, ... is removed
        patient_numbers[i] = re.sub('[^0-9]', '', image_path.stem.split("_")[0]) 
        diagnose_grouped.append(ground_truth[ground_truth["sample"]==patient_numbers[i]]["diagnose_grouped"].values[0])
    unique_patient_numbers = np.unique(patient_numbers)
    merged_info = pd.DataFrame(np.array([image_paths, patient_numbers, diagnose_grouped]).transpose(), columns=["path", "patient_number", "diagnose_grouped"])
    merged_info["dataset"]= dataset
    return merged_info


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
        result_patient = pd.DataFrame([res_p.iloc[0][["patient", "gt_ind", "gt_name", "split"]]], columns=["patient", "gt_ind", "gt_name", "split"])
        result_patient = pd.concat([result_patient.reset_index(), pd.DataFrame([[pred_ind, *n_classifications]], columns=["pred_ind", *results_classes])], axis=1, sort=False)
        result_patient.drop(columns=["index"], inplace=True)
        results_patient = results_patient.append(result_patient, ignore_index=True)
    results_patient["right_pred"] = results_patient["gt_ind"]==results_patient["pred_ind"]
    return results_patient


class GeneralizationDataset(Dataset):
    def __init__(self, dataset_info, transform=None):
        self.dataset_info = dataset_info
        self.transform = transform
        self.class_to_idx = {"non-PTC-like": 0, "PTC-like": 1}

    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, index):
        img = Image.open(self.dataset_info["path"][index])
        target = self.dataset_info["diagnose_grouped"][index]
        # target needs to be transformed to suite pytorch trained models (0 => non-PTC-like, 1 => PTC-like)
        if target == "PTC-like":
            target = 1
        else:
            target = 0
        if self.transform is not None:
            data = self.transform(image=np.array(img), target=target) #window not needed here
        return {"image": data["image"], "target": data["target"], "path": str(self.dataset_info["path"][index]), "window": 0}


if direction == "TtoN":
    # get original resolution of the Tharun and Thompson dataset (40x)
    T_paths = get_image_paths(T_folder_40x)
    T_ground_truth = pd.read_excel(T_folder_40x.joinpath("ground_truth.xlsx"), engine="openpyxl")
    T_data = merge_path_gt(T_paths, T_ground_truth, dataset="Tharun and Thompson")
    
    # get upscaled Nikiforov dataset (matching the T&T dataset)
    N_paths = get_image_paths(N_folder_40x)
    N_ground_truth = pd.read_excel(N_folder_40x.joinpath("ground_truth.xlsx"), engine="openpyxl")
    N_data = merge_path_gt(N_paths, N_ground_truth, dataset="Nikiforov")

    networks_train = networks_T
    test_data = N_data
    save_path = Path(__file__).parent.joinpath("dlc_TtoN_generalization.xlsx")
else:
    # get original Nikiforov dataset (20x)
    N_paths = get_image_paths(N_folder_20x)
    N_ground_truth = pd.read_excel(N_folder_20x.joinpath("ground_truth.xlsx"), engine="openpyxl")
    N_data = merge_path_gt(N_paths, N_ground_truth, dataset="Nikiforov")    

    # get downscaled Tharun and Thompson dataset (matching the Nikiforv dataset)
    T_paths = get_image_paths(T_folder_20x)
    T_ground_truth = pd.read_excel(T_folder_20x.joinpath("ground_truth.xlsx"), engine="openpyxl")
    T_data = merge_path_gt(T_paths, T_ground_truth, dataset="Tharun and Thompson")

    networks_train = networks_N
    test_data = T_data
    save_path = Path(__file__).parent.joinpath("dlc_NtoT_generalization.xlsx")

results = pd.DataFrame(columns=["split", "patient", "gt_ind", "pred_ind", "gt_name", 1, 0, "right_pred"])
for idx, network_path in enumerate(networks_train):
    model = ClassificationModule.load_from_checkpoint(checkpoint_path=str(network_path))
    trainer = pl.Trainer(gpus=1, logger=None) #, progress_bar_refresh_rate=0
    trans = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    dataset = GeneralizationDataset(test_data, transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=24)


    trainer.test(model, test_dataloaders=dataloader)
    results_split = model.results
    results_split["split"] = idx
    results_split = results_image_to_patient(results_split)
    results = results.append(results_split, ignore_index=True)

results.to_excel(save_path, index=False)
print("End of Evaluation")