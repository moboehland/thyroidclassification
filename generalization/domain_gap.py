from albumentations.augmentations.transforms import Normalize
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import numpy as np
import re
import umap
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

""" Programm to evaluate if there is a significant domain gap between two datasets
To see if there is a domain gap, a pretrained Resnet50 is used to extract features from both datasets and UMAP is used for unsupervised clustering. When distinct clusters for both datasets
are formed, there is a domain gap present.    
The domain gap can be evaluated for native Tharun and Thompson and upscaled Nikiforov as well as native Nikiforov and downscaled Tharun and Thompson.
Furthermore, it can be evaluated on the native version on both datasets.
"""

native_dataset = "N"  # T, N or both


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


def draw_scatter(data, scatter_path, target):
    umap_plt = sns.scatterplot(data=data, x="UMAP 1", y="UMAP 2", hue=target)
    #umap_plt.set(title="Umap thyroid tumor")
    umap_fig = umap_plt.get_figure()
    umap_fig.savefig(scatter_path, bbox_inches="tight")
    plt.close(umap_fig)

def apply_umap(measures, features, native_dataset, target="target", hparams={}):
    # only keep patient, feature selection, diagnose
    measures_umap = measures.copy()
    scaler = StandardScaler()
    measures_umap.reset_index(inplace=True)
    measures_umap[features] = pd.DataFrame(scaler.fit_transform(measures_umap[features]), columns=features)
    reducer = umap.UMAP(**hparams)
    embedding = reducer.fit_transform(measures_umap[features].values)
    embedding = pd.DataFrame(list(zip(embedding[:,0], embedding[:,1], measures_umap[target], measures_umap["path"])), columns=["UMAP 1", "UMAP 2", target, "path"])
    draw_scatter(embedding, Path(__file__).parent.joinpath("domain_gap_"+target+"_native"+native_dataset+"_umap.png"), target)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class DomainGapDataset(Dataset):
    def __init__(self, dataset_info, transform=None):
        self.dataset_info = dataset_info
        self.transform = transform

    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, index):
        img = Image.open(self.dataset_info["path"][index])
        target = self.dataset_info["diagnose_grouped"][index]
        if self.transform is not None:
            data = self.transform(image=np.array(img), target= target)
        return data

def extract_dl_features(image_info, features_path):
    trans = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])
    dataset = DomainGapDataset(image_info, transform=trans)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    net = models.resnet50(pretrained=True)
    net.fc = Identity()
    net.to(torch.device("cuda"))
    net.eval()

    dl_features = np.zeros([len(loader), 2048])
    with torch.no_grad():
            for step, item in enumerate(loader):
                item["image"]= item["image"].to(torch.device("cuda"))
                features = net(item["image"]).cpu().numpy()
                dl_features[step,:] = features.squeeze()

    columns = ["feature_"+str(i) for i in range(dl_features.shape[1])]
    dl_features_pd = pd.DataFrame(data=dl_features, columns=columns)
    dl_features = pd.concat([image_info, dl_features_pd], axis=1)
    dl_features.to_hdf(features_path, key="dl_features", mode="w")
    return dl_features

if native_dataset == "T":
    # get original resolution of the Tharun and Thompson dataset (40x)
    T_paths = get_image_paths(T_folder_40x)
    T_ground_truth = pd.read_excel(T_folder_40x.joinpath("ground_truth.xlsx"), engine="openpyxl")
    T_data = merge_path_gt(T_paths, T_ground_truth, dataset="Tharun and Thompson")
    
    # get upscaled Nikiforov dataset (matching the T&T dataset)
    N_paths = get_image_paths(N_folder_40x)
    N_ground_truth = pd.read_excel(N_folder_40x.joinpath("ground_truth.xlsx"), engine="openpyxl")
    N_data = merge_path_gt(N_paths, N_ground_truth, dataset="Nikiforov")

    image_info = pd.concat([T_data, N_data], ignore_index=True)
    features_path = Path(__file__).parent.joinpath("dl_features_nativeT.h5")
elif native_dataset == "N":
    # get original Nikiforov dataset (20x)
    N_paths = get_image_paths(N_folder_20x)
    N_ground_truth = pd.read_excel(N_folder_20x.joinpath("ground_truth.xlsx"), engine="openpyxl")
    N_data = merge_path_gt(N_paths, N_ground_truth, dataset="Nikiforov")    

    # get downscaled Tharun and Thompson dataset (matching the Nikiforv dataset)
    T_paths = get_image_paths(T_folder_20x)
    T_ground_truth = pd.read_excel(T_folder_20x.joinpath("ground_truth.xlsx"), engine="openpyxl")
    T_data = merge_path_gt(T_paths, T_ground_truth, dataset="Tharun and Thompson")

    image_info = pd.concat([T_data, N_data], ignore_index=True)
    features_path = Path(__file__).parent.joinpath("dl_features_nativeN.h5")
else:
    # get original Nikiforov dataset (20x)
    N_paths = get_image_paths(N_folder_20x)
    N_ground_truth = pd.read_excel(N_folder_20x.joinpath("ground_truth.xlsx"), engine="openpyxl")
    N_data = merge_path_gt(N_paths, N_ground_truth, dataset="Nikiforov")    

    # get original resolution of the Tharun and Thompson dataset (40x)
    T_paths = get_image_paths(T_folder_40x)
    T_ground_truth = pd.read_excel(T_folder_40x.joinpath("ground_truth.xlsx"), engine="openpyxl")
    T_data = merge_path_gt(T_paths, T_ground_truth, dataset="Tharun and Thompson")

    image_info = pd.concat([T_data, N_data], ignore_index=True)
    features_path = Path(__file__).parent.joinpath("dl_features_nativeBoth.h5")

try:
    dl_features = pd.read_hdf(features_path, mode="r+", key="dl_features")
except FileNotFoundError:
    dl_features = extract_dl_features(image_info, features_path)

#extract all feature column names
features = [string for string in dl_features.columns if "feature" in string]
apply_umap(dl_features, features, native_dataset, target="dataset")

print("test")