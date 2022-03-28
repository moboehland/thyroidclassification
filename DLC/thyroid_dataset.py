from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import numpy as np
import os
from PIL import Image

class ThyroidDataset(Dataset):
    def __init__(self, image_folder, split, transform=None, class_imbalance_threshold=100, sliding_window=0):
        """ Pytorch dataset for Thyroid classification with defined splits

        Args:
            image_folder (PosixPath): Folder to the images.
            split (list of tuples): Split and classification used for images. Tuples like (img_name, class)
            transform (transforms, optional): Augmentations used on images.
            class_imbalance_threshold (int, optional): Max percentage, number of samples of classes in dataset can differ.
            sliding_window (int, optional): Size for x and y for overlapping sliding windows created for training.
        """
        super().__init__() # not necessary since Dataset has no own init
        self.image_folder = image_folder
        self.split = split
        self.transform = transform
        self.class_imbalance_threshold = class_imbalance_threshold
        self.sliding_window = sliding_window
        self.class_to_idx = self.create_class_idx(self.split)
        self.samples = self.make_dataset(self.image_folder, self.class_to_idx, self.split)
        
        if self.class_imbalance_threshold < 100:
            self.weight_classes()
            
        self.sliding_window = 512
        if self.sliding_window > 0:
            self.create_slides()
        
    def create_class_idx(self, split):
        unique_classes = []
        for img_class in split:
            if img_class[1] not in unique_classes:
                unique_classes = unique_classes + [img_class[1]]
        class_idx = dict(zip(unique_classes, np.arange(0,len(unique_classes))))
        print(f"Unique classes: {class_idx}")
        return class_idx
   
    def make_dataset(self, dir, class_to_idx, split):
        images = []
        for img_class in split:
            # [0,0,0] is window (x_start, y_start, window size) whole image, sliding_windows are created later if needed
            img_path = dir.resolve().joinpath(img_class[0])
            if not img_path.exists():
                raise (RuntimeError(f"{img_path} created from split and {dir} not found in {dir}."))
            item = (img_path, class_to_idx[img_class[1]], np.array([0,0,0]))  
            images.append(item)
        return images

    def weight_classes(self):
        sample_classes = [sample[1] for sample in self.samples]
        sample_classes_unique, sample_counts_initial = np.unique(sample_classes, return_counts=True)

        sample_counts = sample_counts_initial.copy()
        max_value = sample_counts.max()
        min_value = sample_counts.min()
        while np.round((1-min_value/max_value)*100) > self.class_imbalance_threshold:
            for idx, sample in enumerate(sample_counts):
                sample_counts[idx] = int(np.round(max_value/sample_counts_initial[idx]))*sample_counts_initial[idx]
            if min_value == sample_counts.min():  # nothing changed add initial samples to minimal samples
                sample_counts[sample_counts.argmin()] = sample_counts[sample_counts.argmin()]+sample_counts_initial[sample_counts.argmin()]
            max_value = sample_counts.max()
            min_value = sample_counts.min()

        multiplier = sample_counts/sample_counts_initial
        samples_new = []
        for sample_idx in sample_classes_unique:
            samples_new = samples_new + [sample for sample in self.samples if sample[1] == sample_idx]*int(multiplier[sample_idx])

        self.samples = samples_new
        return

    def create_slides(self):
        samples_new = []
        for sample in self.samples:
            img = Image.open(sample[0])
            shape = img.size
            windows_x = self.calc_windows_start(self.sliding_window, shape[0])
            windows_y = self.calc_windows_start(self.sliding_window, shape[1])

            for window_x in windows_x:
                for window_y in windows_y:
                    samples_new.append((*sample[0:2], np.array([window_x, window_y, self.sliding_window])))
        self.samples=samples_new

    def calc_windows_start(self, window_len, img_len):
        n_windows = int(np.ceil(img_len/window_len))
        overlap_windows = n_windows - 1
        pix_overlap = n_windows*window_len-img_len
        overlap_per_window = pix_overlap/(n_windows-1)  # - because first window cannot overlap
        min_overlap = np.floor(overlap_per_window)
        overlap_remaining = int(pix_overlap-min_overlap*overlap_windows)
        overlap_remaining_pos = np.linspace(1,n_windows, overlap_remaining, dtype=int)
        windows = []
        windows.append(0)
        for i in np.arange(start=1, stop=n_windows):
            start = windows[i-1]+window_len
            start = start - min_overlap
            if i in overlap_remaining_pos:
                start = start -1
            windows.append(start.astype(int))  # astype because first element is int and start for other elements would be float this would break dataloader
        return windows

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path, window) where target is class_index of the target class.
        """
        path, target, window = self.samples[index]
        sample = Image.open(path)
        sample = sample.convert('RGB')
        if self.sliding_window > 0:  # if sliding window is activate, crop image to window
            sample = sample.crop((window[0], window[1], window[0]+window[2], window[1]+window[2]))
        if self.transform is not None:
            sample = self.transform(image=np.array(sample), target=target)

        return {"image": sample["image"], "target": sample["target"], "path": str(path), "window": window}
