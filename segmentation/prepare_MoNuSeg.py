import numpy as np
import cv2
from scipy import ndimage
from pathlib import Path
from PIL import Image
import numpy as np

""" Steps to prepare the training data (MoNuSeg Challenge data) for the segmentation network
1) Download training and test data from https://monuseg.grand-challenge.org/Data/
2) Unpack train data to folder MoNuSeg/train and test data to folder MoNuSeg/val
3) In MoNuSeg/val create folder "Tissue Images" and "Annotations" and put *.tif images into "Tissue Images" and *.xml* files into "Annotations
   Structure in MoNuSeg/val needs to match structure in MoNuSeg/train
4) Run he_to_instance_mask.m in matlab
5) Run prepare_train_data (this script) with the path to the MoNuSeg folder (folder_monuseg).
6) See if files _adapted_border.tif and _binary.tif have been created for each image in the Annotations folder. 
"""

# Fill in path to the monuseg main folder if you run this script from the segmentation folder the path needs to be ../datasets/MoNuSeg
folder_monuseg = Path.cwd().joinpath("datasets/MoNuSeg")


def adapted_border_label(label):
    """ Adapted border label image creation.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :return: Adapted border label image.
    """

    kernel = np.ones(shape=(3, 3), dtype=np.uint8)

    label_bin = label>0

    boundary = cv2.Canny(label.astype(np.uint8), 1, 1) > 0  
    # cv2.Canny can only handel np.uint8 cells with label x*256 are set to zero by astype(np.uint8) and border errors occure
    # for future projects this can be fixed by using skimage.feature.canny like canny(label, sigma=0, low_threshold=0.1, high_threshold=0.1)
    # skimage canny can handle unint16 directly
    # set sigma to zero to disable gaussian smoothing since it is not needed (all edges need to be detected)

    border = cv2.Canny(label_bin.astype(np.uint8), 1, 1) > 0
    border = boundary ^ border

    border_adapted = ndimage.binary_dilation(border.astype(np.uint8), kernel)
    cell_adapted = ndimage.binary_erosion(label_bin.astype(np.uint8), kernel)

    border_adapted = ndimage.binary_closing(border_adapted, kernel)
    label_adapted_border = (np.maximum(cell_adapted, 2 * border_adapted)).astype(np.uint8)

    return label_adapted_border


folder_train_annotations = folder_monuseg.joinpath("train/Annotations")
folder_val_annotations = folder_monuseg.joinpath("val/Annotations")

paths_train_annotations = sorted(folder_train_annotations.glob('*_instance.png'))
paths_val_annotations = sorted(folder_val_annotations.glob('*_instance.png'))
paths_annotations = paths_train_annotations+paths_val_annotations

for path_annotation in paths_annotations:
    annotation = np.asarray(Image.open(path_annotation)).astype(np.uint16)
    annotation_binary = (annotation > 0).astype(np.uint8)
    annotation_adapted_border = adapted_border_label(annotation)

    path_annotation = path_annotation.parent.joinpath(path_annotation.stem+".tif")
    path_annotation_binary = path_annotation.parent.joinpath(path_annotation.stem[:-9]+"_binary.tif")
    path_annotation_adapted_border = path_annotation.parent.joinpath(path_annotation.stem[:-9]+"_adapted_border.tif")

    # zero pad to be able to pass whole image through u-net
    annotation = np.pad(annotation, pad_width=((0,24),(0,24)))
    annotation_binary = np.pad(annotation_binary, pad_width=((0,24),(0,24)))
    annotation_adapted_border = np.pad(annotation_adapted_border, pad_width=((0,24),(0,24)))

    Image.fromarray(annotation).save(path_annotation)
    Image.fromarray(annotation_binary*255).save(path_annotation_binary)
    Image.fromarray(annotation_adapted_border).save(path_annotation_adapted_border)

# zero pad also images with white (255,255,255)
folder_train_images = folder_monuseg.joinpath("train/Tissue Images")
folder_val_images = folder_monuseg.joinpath("val/Tissue Images")
paths_train_images = sorted(folder_train_images.glob('*.tif'))
paths_val_images = sorted(folder_val_images.glob('*.tif'))
paths_images = paths_train_images+paths_val_images

for path_image in paths_images:
    image = np.asarray(Image.open(path_image))
    pad_width = 1024-image.shape[0]
    image = np.pad(image, pad_width=((0,pad_width),(0,pad_width),(0,0)), constant_values=255)
    Image.fromarray(image).save(path_image)
