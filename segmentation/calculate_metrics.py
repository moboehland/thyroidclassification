from pathlib import Path
from customs.metrics import metric_collection, calc_metrics
from PIL import Image
import numpy as np
import json


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',
                  '.TIFF', '.tif', '.TIF']

folder_pred = Path("/srv/user/boehland/Luebeck/Lars/Data/Benchmarks/MoNuSeg_GrandChallenge/Data/original_pad/val/images/results/2048_adam_iter3/")
folder_groundtruth = Path("/srv/user/boehland/Luebeck/Lars/Data/Benchmarks/MoNuSeg_GrandChallenge/Data/original_pad/val/annotations/")

def get_image_paths(folder):
    image_paths = []
    for file in folder.iterdir():
        if any(file.suffix == extension for extension in IMG_EXTENSIONS):
            image_paths.append(file)
    return image_paths

image_paths_pred = get_image_paths(folder_pred)
image_paths_groundtruth = get_image_paths(folder_groundtruth)

res = []
for i, path in enumerate(image_paths_pred):
    if "instance.png" in path.name:
        image_path_groundtruth = folder_groundtruth.joinpath(path.stem+".tif")
        pred = np.asarray(Image.open(path))
        gt = np.asarray(Image.open(image_path_groundtruth))
        res.append(metric_collection(pred, gt))
metrics = calc_metrics(res)
res_path = folder_pred.joinpath("metrics.json")
with open(res_path, 'w', encoding='utf-8') as outfile:
    json.dump(metrics, outfile, ensure_ascii=False, indent=2)

print("test")









