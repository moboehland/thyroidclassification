import numpy as np
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import sys
sys.path.append(str(Path(__file__).parents[1].joinpath("FBC")))
from utils import data_loader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import ParameterGrid
import xgboost as xgb
from sklearn.decomposition import PCA
from skrebate import SURF
import json

 





def aggregate_to_patient(measures, feature_selection, diagnose_selection):
    measures_mean = measures[measures["diagnose_grouped"].isin(diagnose_selection)].groupby(['sample', 'diagnose_grouped']).mean()
    measures_mean = measures_mean.add_suffix("_mean")
    measures_std = measures[measures["diagnose_grouped"].isin(diagnose_selection)].groupby(['sample', 'diagnose_grouped']).std()
    measures_std = measures_std.add_suffix("_std")
    feature_selection_mean = [x+"_mean" for x in feature_selection]
    feature_selection_std = [x+"_std" for x in feature_selection]

    measures_aggr = pd.concat([measures_mean, measures_std], axis=1)
    measures_aggr.reset_index(inplace=True)
    feature_selection_agg = feature_selection_mean + feature_selection_std
    return measures_aggr, feature_selection_agg


def create_detailed_results_filename(classifier, n_features, ending):
    hp = classifier.data_preproc_hparams
    qt = "QTon" if hp.quantile_transform else "QToff"
    scale = "SCALE"+hp.scaler
    n_feats = "NFEATURES"+str(n_features)
    fsm = "FSM"+hp.feature_selection_method
    class_name = str(type(classifier.Classifier())).split('.')[-1][:-2]
    results_filename = Path(class_name+"_"+qt+"_"+scale+"_"+fsm+"_"+n_feats+ending)
    return results_filename


def save_split_to_file(measures, file_path):
    # Extract splits here: after first split break
    patients_train = list(map(int, measures[measures["split"]=="train"]["sample"]))
    patients_val = list(map(int, measures[measures["split"]=="val"]["sample"]))
    patients_test = list(map(int, measures[measures["split"]=="test"]["sample"]))

    file_path.parent.mkdir(exist_ok=True, parents=True)
    split_patients = {"train":patients_train, "val": patients_val, "test": patients_test}
    with open(file_path, 'w', encoding="utf-8") as outfile:
        json.dump(split_patients, outfile, ensure_ascii=False, indent=2)

def group_data(df, groups):
    for key, value in groups.items():
        print(f"key {key}, value {value}")
        rep_inner = {}
        for v in value:
            rep_inner[v] = key
        rep = {"diagnose_grouped": rep_inner}
        df.replace(rep, inplace=True)
    return df


def main(hparams):
    measures_T = data_loader(hparams.features_folder_T)  # , patients=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20]
    group_data(measures_T, hparams.diagnose_groups_T)
    measures_T, feature_selection = aggregate_to_patient(measures_T, hparams.feature_selection_initial, hparams.diagnose_selection)

    measures_N = data_loader(hparams.features_folder_N)  # , patients=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20]
    group_data(measures_N, hparams.diagnose_groups_T)
    measures_N, feature_selection = aggregate_to_patient(measures_N, hparams.feature_selection_initial, hparams.diagnose_selection)

  
    # only keep patient, feature selection, diagnose_grouped
    measures_N = measures_N[["sample", "diagnose_grouped", *feature_selection]]
    measures_T = measures_T[["sample", "diagnose_grouped", *feature_selection]]

    if hparams.direction == "TtoN":
        measures_train = measures_T
        measures_test = measures_N
    else:
        measures_train = measures_N
        measures_test = measures_T
    
    # standard scale data
    scaler = StandardScaler()
    measures_train[feature_selection] = pd.DataFrame(scaler.fit_transform(measures_train[feature_selection]), columns=feature_selection)
    measures_test[feature_selection] = pd.DataFrame(scaler.transform(measures_test[feature_selection]), columns=feature_selection)

    clf = SVC(random_state=0)
    clf.fit(measures_train[feature_selection], measures_train["diagnose_grouped"])
    train_accuracy = clf.score(measures_train[feature_selection], measures_train["diagnose_grouped"])
    test_accuracy = clf.score(measures_test[feature_selection], measures_test["diagnose_grouped"])
    print(f'Trained: {hparams.direction} with {train_accuracy} train accuracy and {test_accuracy} test accuracy.')

    print("Classification done")



if __name__ == '__main__':
    features_folder_N = Path(__file__).parent.joinpath("..", "datasets", "Nikiforov_upscale2x", "results", "features").resolve()
    features_folder_T = Path(__file__).parent.joinpath("..", "datasets", "TharunThompson", "results", "features").resolve()
    diagnose_groups_T = {"negative": ["non-PTC-like"], "positive": ["PTC-like"]}
    diagnose_groups_N = {"negative": ["non-PTC-like"], "positive": ["PTC-like"]}
    # class labels need to be renamed to exactly negative and positive, otherwise results are not reproduceable for the svm
    # see https://github.com/scikit-learn/scikit-learn/issues/11263

    feature_selection_initial =     ["area", "eccentricity", "perimeter", "solidity",
                                    "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std", "gray_equal_mean", "gray_equal_std",
                                    "neighbor_distance",
                                    "neighbours_in_radius_factor_3", "neighbours_in_radius_factor_5", "neighbours_in_radius_factor_7", "neighbours_in_radius_factor_9",
                                    "neighbours_in_radius_factor_15", "neighbours_in_radius_factor_20", "neighbours_in_radius_factor_25", "neighbours_in_radius_factor_30",
                                    "gray_equal_shannon_entropy",
                                    'energy_angle_merge_1', 'homogeneity_angle_merge_1', 'dissimilarity_angle_merge_1', 'correlation_angle_merge_1',
                                    'energy_angle_merge_2', 'homogeneity_angle_merge_2', 'dissimilarity_angle_merge_2', 'correlation_angle_merge_2',
                                    'ratio_gray_border_mean_gray_middle_mean', 'ratio_gray_border_mean_gray_center_mean', 'ratio_gray_middle_mean_gray_center_mean',
                                    'ratio_gray_border_std_gray_middle_std', 'ratio_gray_border_std_gray_center_std', 'ratio_gray_middle_std_gray_center_std']    

 
    parser = ArgumentParser()
    parser.add_argument("--features_folder", default="T", type=str, help="path to features for each patient if T or N default folders are used")
    parser.add_argument("--diagnose_selection", type=str, nargs="+", default=["negative", "positive"])
    parser.add_argument("--feature_selection_method", type=str, default="none", help="Method for feature_selection (sfs, rfe, chi2, none)")
    parser.add_argument("--feature_selection_num_features", type=int, default=25, help="Number of features to select, if feature_selection_method is not empty")
    parser.add_argument("--results_folder_name", type=str, default="test", help="Folder name for all files written.")
    parser.add_argument("--quantile_transform", default=False, action='store_true', help="Use quantile transformation on dataset.")
    parser.add_argument("--scaler", type=str, default="none", help="Scaler [std] (std: standard scaler)")
    parser.add_argument("--direction", type=str, default="NtoT", help="Direction of the generalization test.")
    hparams = parser.parse_args()

    hparams.feature_selection_initial = feature_selection_initial

    hparams.features_folder_T = features_folder_T
    hparams.diagnose_groups_T = diagnose_groups_T

    hparams.features_folder_N = features_folder_N
    hparams.diagnose_groups_N = diagnose_groups_N

    hparams.feature_selection_num_features = np.arange(1, hparams.feature_selection_num_features+1)
    main(hparams)
