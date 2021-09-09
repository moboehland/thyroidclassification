from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from utils import find_corresponding_labels
from sklearn.preprocessing import OneHotEncoder
import torch


class HistopathologyDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=lambda x: x, dataset_iter=1):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.transforms = transforms
        self.dataset_iter = dataset_iter
        if self.mode == "test":
            # If images to infere are not present as *.tif please change the ending accordingly
            self.img_paths = sorted(self.root_dir.glob('*.tif'))
        elif self.mode == "train":
            img_paths = sorted(self.root_dir.joinpath(self.mode, "Tissue Images").glob('*.tif'))
            self.img_paths = []
            for i in range(0, self.dataset_iter):
                self.img_paths = [*self.img_paths, *img_paths]
            self.labels_folder = self.root_dir.joinpath(self.mode, "Annotations")
            self.label_border_seed_paths = find_corresponding_labels(self.img_paths, self.labels_folder, "_adapted_border.tif")
            self.label_cells_paths = find_corresponding_labels(self.img_paths, self.labels_folder, "_binary.tif")
        elif self.mode == "val":
            self.img_paths = sorted(self.root_dir.joinpath(self.mode, "Tissue Images").glob('*.tif')) # self.mode #.tif/png
            self.labels_folder = self.root_dir.joinpath(self.mode, "Annotations")              # self.mode
            self.label_border_seed_paths = find_corresponding_labels(self.img_paths, self.labels_folder, "_adapted_border.tif")
            self.label_cells_paths = find_corresponding_labels(self.img_paths, self.labels_folder, "_binary.tif")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = np.asarray(Image.open(img_path)).astype(np.uint8)


        if self.mode == "test":
            sample = {'image': img, 'id': img_path.stem}
            sample["image"] = self.transforms(image=sample["image"])["image"].to(torch.float)
            return sample

        label_border_seed_path = self.label_border_seed_paths[idx]
        label_border_seed = np.asarray(Image.open(label_border_seed_path)).astype(np.uint8)
        enc = OneHotEncoder(sparse=False, categories='auto')
        one_hot = enc.fit_transform(label_border_seed.reshape(-1,1))
        label_seed = one_hot[:,1].reshape(label_border_seed.shape)
        label_border = one_hot[:,2].reshape(label_border_seed.shape)


        label_cells_path = self.label_cells_paths[idx]
        label_cells = np.asarray(Image.open(label_cells_path)).astype(np.uint8)


        sample = {'image': img,
                  'label_border': label_border,
                  'label_seed': label_seed,
                  'label_cell': label_cells,
                  'id': img_path.stem}

        trans = self.transforms(image=sample["image"], label_border=sample["label_border"], label_seed=sample["label_seed"], label_cell=sample["label_cell"])
        trans["image"] = trans["image"].to(torch.float)
        trans["id"] = sample["id"]
        return trans
