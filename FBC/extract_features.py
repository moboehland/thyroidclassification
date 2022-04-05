import numpy as np
from pathlib import Path
from PIL import Image
from numpy.lib.arraysetops import unique
import skimage.morphology as morphology
from skimage.measure import regionprops, regionprops_table, shannon_entropy
import pandas as pd
from scipy.spatial import distance
import re
from skimage.feature import greycomatrix, greycoprops
from scipy.ndimage.morphology import distance_transform_edt

""" Extract features from a dataset

Specify the path to the dataset in dataset_folder
"""

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',
                  '.TIFF', '.tif', '.TIF']

dataset_folder = Path(__file__).parent.joinpath("..", "datasets", "Nikiforov_upscale2x").resolve()
instance_folder = dataset_folder.joinpath("results/segmentation")
image_folder = dataset_folder.joinpath("")
ground_truth_path = dataset_folder.joinpath("ground_truth.xlsx")
image_datatype = "tif"
cell_size_min = 0.5
cell_size_max = 2
ellipse_ratio = 1/2.5


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


def get_patient_numbers(image_paths):
    patient_numbers = np.zeros(len(image_paths))
    image_paths = np.asarray(image_paths)
    for i, image_path in enumerate(image_paths):
        # if patient has multiple images e.g. 1a, 1b, ... a,b, ... is removed
        patient_numbers[i] = re.sub('[^0-9]', '', image_path.stem.split("_")[0]) 
    unique_patient_numbers = np.unique(patient_numbers)
    return unique_patient_numbers, patient_numbers


def get_distance_intensity_features(cell, cell_seg_binary):
    cell_seg_binary_pad = np.pad(cell_seg_binary, (1), mode="constant")  # add padding to get distance of 1 for border cell pixels
    cell_seg_dst = distance_transform_edt(cell_seg_binary_pad)[1:-1, 1:-1]  # remove padding directly
    
    cell_seg_dst = (cell_seg_dst*255/cell_seg_dst.max()).astype("uint8")
    
    cell_seg_dst = cell_seg_dst
    p_33 = np.percentile(cell_seg_dst[cell_seg_dst>0], 33)
    p_66 = np.percentile(cell_seg_dst[cell_seg_dst>0], 66)
    cell_distance_map = np.zeros_like(cell_seg_dst, dtype="uint8")
    cell_distance_map[((cell_seg_dst<=p_33) & (cell_seg_dst>0))] = 1
    cell_distance_map[((cell_seg_dst<=p_66) & (cell_seg_dst>p_33))] = 2
    cell_distance_map[(cell_seg_dst>p_66)] = 3

    gray_border = cell[cell_distance_map==1]
    gray_middle = cell[cell_distance_map==2]
    gray_center = cell[cell_distance_map==3]
    gray_border_mean = gray_border.mean()
    gray_middle_mean = gray_middle.mean()
    gray_center_mean = gray_center.mean()
    gray_border_std = gray_border.std()
    gray_middle_std = gray_middle.std()
    gray_center_std = gray_center.std()
    return np.array([gray_border_mean, gray_middle_mean, gray_center_mean, gray_border_std, gray_middle_std, gray_center_std])


def create_ratio_features(measures, combinations=[["gray_border_mean", "gray_middle_mean", "gray_center_mean"], ["gray_border_std", "gray_middle_std", "gray_center_std"]]):
    features_new = []
    for combination in combinations:
        for i in range(len(combination)):
            for j in np.arange(i+1,len(combination)):
                    feature_name = "ratio_"+combination[i]+"_"+combination[j]
                    measures[feature_name] = measures[combination[i]]/(measures[combination[i]]+measures[combination[j]])


def extract_cell_features(image, label, measures, image_path, instance_path):
    # sanity check
    r_mean, r_std, g_mean, g_std, b_mean, b_std = [], [], [], [], [], []
    gray_equal_shannon_entropy = []
    gray_equal_mean, gray_equal_std, gray_perceptual_mean, gray_perceptual_std = [], [], [], []
    if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
        raise RuntimeError("Image and Instance shapes do match!")

    glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] 
    glcm_distances = [1,2]
    glcm_properties = ['dissimilarity', 'correlation', 'energy', 'homogeneity']
    glcm_angles_str = list(map(str, np.tile((np.array(glcm_angles)*180/np.pi).astype("uint8"), len(glcm_distances)*len(glcm_properties))))
    glcm_underscores = ["_"]*len(glcm_angles)*len(glcm_distances)*len(glcm_properties)
    glcm_distances_str = list(map(str, np.repeat(np.array(glcm_distances).astype("uint8"), len(glcm_angles))))*len(glcm_properties) 
    glcm_properties_str = np.repeat(glcm_properties, len(glcm_angles)*len(glcm_distances))
    glcm_feature_names = [p+u+a+u+d for p, a,u,d in zip(glcm_properties_str, glcm_angles_str, glcm_underscores, glcm_distances_str)] 
    glcm_feature_names = glcm_feature_names+[p+"_angle_merge_"+d for d,p in zip(list(map(str, np.tile(glcm_distances, len(glcm_properties)).astype("uint8"))), np.repeat(glcm_properties, len(glcm_distances)))]
    glcm_features = []
    dist_intensity_features = []

    distance_intensity_feature_names = ["gray_border_mean", "gray_middle_mean", "gray_center_mean","gray_border_std", "gray_middle_std", "gray_center_std"]

    for idx in range(len(measures)):
        # use iloc for integer location
        coords = measures.iloc[idx]["coords"]
        cell_ids = label[coords[:,0], coords[:,1]]
        if not all(cell_ids == measures.iloc[idx]["cell_label"]):
            raise RuntimeError("Cell extraction error")
        cell = image[coords[:,0], coords[:,1], :]
        r_std.append(cell[:,0].std())
        g_mean.append(cell[:,1].mean())
        g_std.append(cell[:,1].std())
        b_mean.append(cell[:,2].mean())
        r_mean.append(cell[:,0].mean())
        b_std.append(cell[:,2].std())
        gray_perceptual = 0.2125*cell[:,0] + 0.7154*cell[:,1] + 0.0721*cell[:,2]
        gray_perceptual_mean.append(gray_perceptual.mean())
        gray_perceptual_std.append(gray_perceptual.std())
        gray_equal = 0.3333*cell[:,0] + 0.3333*cell[:,1] + 0.3333*cell[:,2]
        gray_equal_mean.append(gray_equal.mean())
        gray_equal_std.append(gray_equal.std())
        gray_equal_shannon_entropy.append(shannon_entropy(gray_equal))
        bbox = [measures.iloc[idx]["bbox-0"], measures.iloc[idx]["bbox-1"], measures.iloc[idx]["bbox-2"], measures.iloc[idx]["bbox-3"]]
        cell_box = image[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
        cell_box_gray = (0.3333*cell_box[...,0] + 0.3333*cell_box[...,1] + 0.3333*cell_box[...,2]).astype("uint8")
        dist_intensity_features.append(get_distance_intensity_features(cell_box_gray, measures.iloc[idx]["image"]))
        glcm_features.append(get_glcm_features(cell_box_gray, angles=glcm_angles, distances=glcm_distances, properties=glcm_properties))
    

    
    measures_glcm = pd.DataFrame(np.array(glcm_features), columns=glcm_feature_names)
    measures_distance_intensity = pd.DataFrame(np.array(dist_intensity_features), columns=distance_intensity_feature_names)
    measures.reset_index(inplace=True)  # to be able to join columns of dfs
    measures = pd.concat([measures, measures_glcm, measures_distance_intensity], axis=1, join="inner")
    measures["gray_equal_shannon_entropy"] = gray_equal_shannon_entropy
    measures["r_mean"] = r_mean
    measures["r_std"] = r_std
    measures["g_mean"] = g_mean
    measures["g_std"] = g_std
    measures["b_mean"] = b_mean
    measures["b_std"] = b_std
    measures["gray_perceptual_mean"] = gray_perceptual_mean
    measures["gray_perceptual_std"] = gray_perceptual_std
    measures["gray_equal_mean"] = gray_equal_mean
    measures["gray_equal_std"] = gray_equal_std
    create_ratio_features(measures)
    return measures


def get_glcm_features(cell, angles = [0, np.pi/4, np.pi/2, 3*np.pi/4], distances = [1,2], properties = ['dissimilarity', 'correlation', 'energy', 'homogeneity']):
    glcm = greycomatrix(cell, 
                    distances=distances,
                    angles=angles,
                    symmetric=True,
                    normed=True)   
    feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])  # ravel => flatten array hstack => list of arrays to array

    # create features for merged angles (sum)
    glcm_angle_sum = np.expand_dims(np.sum(glcm, axis=3), 2)  # 3rd dimension (angles) is needed for greycoprops
    feats_angle_sum = np.hstack([greycoprops(glcm_angle_sum, prop).ravel() for prop in properties])
    return np.concatenate([feats, feats_angle_sum], axis=0)

def extract_distance_nearest_neighbor(df):
    center_points = np.stack((df["centroid-0"].values, df["centroid-1"].values), axis=1)
    distance_map = distance.cdist(center_points, center_points, 'euclidean')
    distance_map_mask = np.ma.masked_equal(distance_map, 0, copy=False)  # Each point has distance 0 to itselt. Remove this values (diagonal axis)
    min_dist = distance_map_mask.min(axis=1).data
    df["neighbor_distance"] = min_dist


def remove_cells_big_small(measures, cell_size_min, cell_size_max):
    """Remove cells smaller and bigger than cell_size_min and cell_size_max times the median cell

    Args:
        measures (pandas dataframe): Dataframe with column name area which is used for removing cells
        cell_size_min (float): Factor used to remove cells smaller median*cell_size_min
        cell_size_max (float): Factor used to remove cells bigger median*cell_size_max
    """
    len_before = len(measures)
    measures.reset_index(inplace=True, drop=True)  # reset index to have each index only one time. This enables removing by index
    area_median = measures["area"].median()
    drop_idx = measures.loc[measures["area"]>(area_median*cell_size_max)].index  # Drop all cells >2*median
    measures.drop(drop_idx, inplace=True)
    drop_idx = measures.loc[measures["area"]<(area_median*cell_size_min)].index  # Drop all cells <0.5*median
    measures.drop(drop_idx, inplace=True)
    print(f"Dropped {len_before-len(measures)} of {len_before} cells by size for image {measures.iloc[0]['image_id']}.")


def remove_elliptic_cells(measures, ellipse_ratio):
    """Remove cells being more elliptic than ellipse_ratio. Cells are removed inplace

    Args:
        measures (pandas dataframe): Dataframe with column name area which is used for removing cells
        ellipse_ratio (float): Ratio of minor_axis_lenght/major_axis_length
    """
    len_before = len(measures)
    measures.reset_index(inplace=True, drop=True)  # reset index to have each index only one time. This enables removing by index
    ellipse_min_div_maj = np.sqrt(1-measures["eccentricity"]**2)  # minor_axis_length/major_axis_length
    drop_idx = measures.loc[ellipse_min_div_maj<ellipse_ratio].index  # Drop all cells with ellipse ratio < given value
    measures.drop(drop_idx, inplace=True)
    print(f"Dropped {len(drop_idx)} of {len_before} cells by eccentricity for image {measures.iloc[0]['image_id']}.")

def get_num_neighbours_in_radius(df, radius_mean_factor):
    center_points = np.stack((df["centroid-0"].values, df["centroid-1"].values), axis=1)
    distance_map = distance.cdist(center_points, center_points, 'euclidean')
    distance_map_mask = np.ma.masked_equal(distance_map, 0, copy=False).mask  # Each point has distance 0 to itselt. Create Mask for this values
    #distance_map[distance_map_mask]= np.NaN  # Set the zeros to NaN
    radius_mean = np.sqrt(df["area"].mean()/(np.pi))
    distance_below_radius = (distance_map<(radius_mean*radius_mean_factor))*~distance_map_mask  # by multiplying with inverse mask, all 0 distances are set to false
    num_neighbours_in_radius = distance_below_radius.sum(axis=1)
    df["neighbours_in_radius_factor_"+str(radius_mean_factor)] = num_neighbours_in_radius

instance_paths = get_image_paths(instance_folder, substring="instance.")
results_folder = dataset_folder.joinpath("results", "features")
results_folder.mkdir(parents=True, exist_ok=True)
unique_patient_numbers, patient_numbers = get_patient_numbers(instance_paths)
unique_patient_numbers = np.sort(unique_patient_numbers)
ground_truth = pd.read_excel(ground_truth_path, engine="openpyxl")

for patient_number in unique_patient_numbers:
    print(f"Evaluating patient number: {patient_number}")
    instance_paths_patient = instance_paths[np.where(patient_numbers == patient_number)]
    measures_patient = pd.DataFrame()
    for instance_path in instance_paths_patient:
        image_path = image_folder.joinpath(instance_path.stem.rsplit("_", 1)[0]+"."+image_datatype)
        instance = np.asarray(Image.open(instance_path))
        image = np.asarray(Image.open(image_path))

        label, num_features = morphology.label(instance, return_num=True)
        measures = pd.DataFrame(regionprops_table(label, properties=['label', 'area', 'eccentricity', 'bbox', 'perimeter', 'solidity', 'image', 'coords', 'centroid']))
        measures.rename(columns={"label":"cell_label"}, inplace=True)
        
        measures["sample"] = patient_number
        measures["image_id"] = str(image_path.stem.split("_")[0][-1])

        remove_cells_big_small(measures, cell_size_min, cell_size_max)
        remove_elliptic_cells(measures, ellipse_ratio)
        
        measures = extract_cell_features(image, label, measures, image_path, instance_path)  # pd.concat => not mutable because new location => return has to be saved
        extract_distance_nearest_neighbor(measures)
        get_num_neighbours_in_radius(measures, 3)
        get_num_neighbours_in_radius(measures, 5)
        get_num_neighbours_in_radius(measures, 7)
        get_num_neighbours_in_radius(measures, 9)
        get_num_neighbours_in_radius(measures, 15)
        get_num_neighbours_in_radius(measures, 20)
        get_num_neighbours_in_radius(measures, 25)
        get_num_neighbours_in_radius(measures, 30)

        measures["diagnose"] = ground_truth.loc[ground_truth["sample"]==patient_number, 'diagnose'].item()
        measures["diagnose_grouped"] = ground_truth.loc[ground_truth["sample"]==patient_number, 'diagnose_grouped'].item()
        # image_id is only meaningfull it the sample has multiple images (001a, 001b, 001c) 
        # otherwise last digit of sample number

        # keep only relevant features and not the ones used only for further calculations
        relevant_features = ["sample", "diagnose", "diagnose_grouped", "image_id", "cell_label", "coords", "area", "eccentricity", "perimeter", "solidity",
                             "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std", "gray_equal_mean", "gray_equal_std",
                             "neighbor_distance",
                             "neighbours_in_radius_factor_3", "neighbours_in_radius_factor_5", "neighbours_in_radius_factor_7", "neighbours_in_radius_factor_9",
                             "neighbours_in_radius_factor_15", "neighbours_in_radius_factor_20", "neighbours_in_radius_factor_25", "neighbours_in_radius_factor_30",
                             "gray_equal_shannon_entropy",
                             'energy_angle_merge_1', 'homogeneity_angle_merge_1', 'dissimilarity_angle_merge_1', 'correlation_angle_merge_1',
                             'energy_angle_merge_2', 'homogeneity_angle_merge_2', 'dissimilarity_angle_merge_2', 'correlation_angle_merge_2',
                             'ratio_gray_border_mean_gray_middle_mean', 'ratio_gray_border_mean_gray_center_mean', 'ratio_gray_middle_mean_gray_center_mean',
                             'ratio_gray_border_std_gray_middle_std', 'ratio_gray_border_std_gray_center_std', 'ratio_gray_middle_std_gray_center_std']  
        measures_patient = measures_patient.append(measures[relevant_features], ignore_index=True)
    
    measures_patient.to_hdf(results_folder.joinpath(str(int(patient_number))+"_features.h5"), key="features", mode="w")

