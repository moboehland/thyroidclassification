# Adaption of the torchvision ImageFolder dataset to support train/test/val splits via split_filenames 
# returns tuple containing sample, target, path


from torchvision.datasets import VisionDataset
from PIL import Image

import os
import os.path
import sys
import numpy as np


class DatasetFolderSplits(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
        class_imbalance_threshold (int): Max percentage, number of samples of classes in dataset can differ.
    """

    def __init__(self, root, loader, split_filenames, transform=None, print_class_to_idx=False, class_imbalance_threshold=100, oversample=1, sliding_window=0):
        super(DatasetFolderSplits, self).__init__(root, transform=transform)
        self.split_filenames = split_filenames
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, self.split_filenames)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + " being in given split filenames."))

        
        self.class_imbalance_threshold = class_imbalance_threshold
        self.sliding_window = sliding_window
        self.oversample = oversample
        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        if print_class_to_idx:
            print(self.class_to_idx)

        if self.class_imbalance_threshold < 100:
            self.weight_classes()

        self.samples = self.samples*self.oversample

        if self.sliding_window > 0:
            self.create_slides()

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


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path, window) where target is class_index of the target class.
        """
        path, target, window = self.samples[index]
        sample = self.loader(path)
        if self.sliding_window > 0:  # if sliding window is activate, crop image to window
            sample = sample.crop((window[0], window[1], window[0]+window[2], window[1]+window[2]))
        if self.transform is not None:
            sample = self.transform(image=np.array(sample), target=target)

        return {"image": sample["image"], "target": sample["target"], "path": path, "window": window}

    def make_dataset(self, dir, class_to_idx, split_filenames):
        images = []
        dir = os.path.expanduser(dir)

        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if fname in split_filenames:
                        item = (path, class_to_idx[target], np.array([0,0,0]))  # [0,0,0] is window (x_start, y_start, window size) at beginning whole image
                        images.append(item)

        return images

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderSplits(DatasetFolderSplits):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.


     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, split_filenames, transform=None, loader=default_loader, print_class_to_idx=False, class_imbalance_threshold=100, oversample=1, sliding_window=0):
        super(ImageFolderSplits, self).__init__(root, loader, split_filenames,
                                                transform=transform,
                                                print_class_to_idx=print_class_to_idx,
                                                class_imbalance_threshold=class_imbalance_threshold,
                                                oversample=oversample,
                                                sliding_window=sliding_window)
        self.imgs = self.samples
