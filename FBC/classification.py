import numpy as np
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
from utils import data_loader, append_df_to_excel, save_confusion_matrix
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


class ClassifierHparamOptimizer():

    def __init__(self, Classifier, class_hparams, data_preproc_hparams, n_splits_outer, n_splits_inner):
        self.Classifier = Classifier
        self.classifier_name = str(type(self.Classifier())).split('.')[-1][:-2]
        self.class_hparams = class_hparams
        self.data_preproc_hparams = data_preproc_hparams
        self.class_hparams_grid = list(ParameterGrid(self.class_hparams))
        self.n_splits_outer = n_splits_outer
        self.n_splits_inner = n_splits_inner
        if self.data_preproc_hparams.feature_selection_method == "none":
            self.data_preproc_hparams.feature_selection_num_features = [-1]
        self.outer_splits = {}
        self.results_columns = ["outer_split", "inner_split", "patients_train", "patients_val", "class_hparams",
                                "n_features", "features", "acc_train", "acc_val"]+list(vars(self.data_preproc_hparams).keys())  # for inner_split = -1 acc_val = acc_test
        self.results = pd.DataFrame(columns=self.results_columns)
        self.data_preproc_hparams_str = {}
        for key, preproc_hparam  in vars(self.data_preproc_hparams).items():
            self.data_preproc_hparams_str[key] = str(preproc_hparam)
        self.pred_test = {}
        for idx, n in np.ndenumerate(self.data_preproc_hparams.feature_selection_num_features):
            self.pred_test[n] = pd.DataFrame(columns=["sample", "gt", "pred"])

    def optimize(self, data_train, data_val, feature_target, feature_selection, outer_split=0, inner_split=-1, additional_training_data=None):
        print(f"Optimizing: {self.classifier_name}, outer split: {outer_split}, inner Split: {inner_split}")
        # Copy data to not change it in original dataframe
        train = data_train.copy()
        val = data_val.copy()

        print(f"Len feature_selection start optimize: {len(feature_selection)}")

        if len(additional_training_data) > 0:  # Additional Training data needs to be dataframe with same columns as X
            train = train.append(additional_training_data.copy())

        train, val = self.scale_transform(train, val, feature_selection)

        features_inner = feature_selection

        if inner_split >= 0:  # optimize hyperparameters (inner splits)
            for idx, p in enumerate(self.class_hparams_grid):
                # Calculate supporting features and afterwards train classifier for each hyperparameter combination
                clf = self.Classifier(**p)

                print(f"Len feature_selection vor sfs: {len(features_inner)}")

                features_inner_selected = self.select_features(clf, train[features_inner], train[feature_target],
                                                               self.data_preproc_hparams.feature_selection_num_features)  # fit classifier with or without feature selection methods (sfs/rfe)

                for idx, n_features in np.ndenumerate(self.data_preproc_hparams.feature_selection_num_features):   # if e.g. sfs is used, multiple number of selected features are calculated at once.
                    features = features_inner_selected[idx[0]]
                    clf.fit(train[features], train[feature_target])
                    acc_train = clf.score(train[features], train[feature_target])
                    acc_val = clf.score(val[features], val[feature_target])
                    pred_right_val = clf.predict(val[features]) == val[feature_target]  # save this to be able to easily calculate mean accuracy over splits with different n_samples later 
                    # Append to results
                    res_data = {"outer_split": outer_split, "inner_split": inner_split, "patients_train": [train["sample"].values],
                                "patients_val": [val["sample"].values],
                                "class_hparams": str(p), "n_features": n_features, "features": str(features),
                                "acc_train": acc_train, "acc_val": acc_val, "pred_right_val": [pred_right_val.values]}
                    res_data.update(self.data_preproc_hparams_str)
                    self.results = self.results.append(pd.DataFrame(res_data, index=[0]), ignore_index=True)
        else:  # optimize outer split with hyperparameters from inner split
            results_split = self.results[self.results["outer_split"]==outer_split]

            # stack pred_right_val for the inner splits and calculate the mean accuracy afterwards. This will calculate the correct accuracy for splits with different numbers of samples
            results_split_grouped = results_split.groupby(["n_features", "class_hparams"])["pred_right_val"].apply(np.hstack).to_frame().reset_index()
            results_split_grouped["val_acc_mean"] = [pred_right_val.mean() for pred_right_val in results_split_grouped["pred_right_val"]]

            for n_features, results in results_split_grouped.groupby(["n_features"]):  # for each number of selected features different classifier hparams are selected
                class_hparams_str = results.iloc[results["val_acc_mean"].argmax()]["class_hparams"]  # get class_hparams for best mean val_acc
                class_hparams = ""
                for combination in self.class_hparams_grid:  # find dict by string from results
                    if str(combination) == class_hparams_str:
                        class_hparams = combination
                        break
                assert class_hparams != "", "Classification hyperparameter combination could not be found!"
                clf = self.Classifier(**class_hparams)
                print(f"Len feature_selection vor sfs: {len(features_inner)}")
                features_selected = self.select_features(clf, train[features_inner], train[feature_target], n_features)  # returns a list with 1 element for scalar n_features
                clf.fit(train[features_selected[0]], train[feature_target])
                acc_train = clf.score(train[features_selected[0]], train[feature_target])
                acc_val = clf.score(val[features_selected[0]], val[feature_target])
                pred_right_train = clf.predict(train[features_selected[0]]) == train[feature_target]
                pred_right_val = clf.predict(val[features_selected[0]]) == val[feature_target]
                # Append to results
                res_data = {"outer_split": outer_split, "inner_split": inner_split, "patients_train": [train["sample"].values],
                            "patients_val": [val["sample"].values],
                            "class_hparams": class_hparams_str, "n_features": n_features, "features": features_selected,
                            "acc_train": acc_train, "acc_val": acc_val, "pred_right_train": [pred_right_train.values], "pred_right_val": [pred_right_val.values]}
                res_data.update(self.data_preproc_hparams_str)
                self.results = self.results.append(pd.DataFrame(res_data, index=[0]), ignore_index=True)
                pred = clf.predict(val[features_selected[0]])
                gt = val[feature_target]
                patients = val["sample"]
                self.pred_test[n_features] = self.pred_test[n_features].append(pd.DataFrame(list(zip(patients, gt, pred)),
                                                                                            columns=["sample", "gt", "pred"]),
                                                                               ignore_index=True)

    def chi2_feature_selection(self, data_train, target, n_features):
        X_norm = MinMaxScaler().fit_transform(data_train)
        chi_selector = SelectKBest(chi2, k=n_features)
        chi_selector.fit(X_norm, target)
        chi_support = chi_selector.get_support()
        chi_features = data_train.loc[:,chi_support].columns.tolist()
        return chi_features

    def select_features(self, clf, X, y, n_features):
        features_support = []
        if self.data_preproc_hparams.feature_selection_method == "rfe":
            rfe_selector = RFE(estimator=clf, n_features_to_select=self.num_automated_selected_features, step=5, verbose=0)
            try:
                rfe_selector.fit(X, y)
                rfe_support = rfe_selector.get_support()  # Mask for features used to classify
            except RuntimeError:
                print(f"Classifier {self.Classifier} has no coef_ and RFE cannot be performed.")
                rfe_support = np.ones(X.shape[1], dtype=bool)  # Use all features to classify
            features_support.append(list(X.columns[rfe_support]))
        elif self.data_preproc_hparams.feature_selection_method == "chi2":
            features_support = []
            for idx, n in np.ndenumerate(n_features):  # ndenumerate is able to enumerate over scalar or vector
                features_support.append(self.chi2_feature_selection(X, y, n))
        elif self.data_preproc_hparams.feature_selection_method == "surf":
            surf_selector = SURF(n_features_to_select=int(np.array(n_features).max()))
            surf_selector.fit(np.array(X),np.array(y))
            feature_importances = surf_selector.feature_importances_
            feature_names = X.columns
            feature_importances_sort_args = np.argsort(feature_importances)  # gets the indices of feature_importances from min to max
            features_support = [list(feature_names[feature_importances_sort_args[-n:]]) for idx, n in np.ndenumerate(n_features)]

        elif self.data_preproc_hparams.feature_selection_method == "sfs":
            sfs_selector = SFS(clf, k_features=int(np.array(n_features).max()), forward=True, floating=False,
                               scoring="accuracy", cv=0, n_jobs=-1)
            sfs_selector.fit(X, y)
            features_support = [list(sfs_selector.subsets_[key]["feature_names"]) for idx, key in np.ndenumerate(n_features)]  # ndenumerate is able to enumerate over scalar
        elif self.data_preproc_hparams.feature_selection_method == "none":
            features_support.append(list(X.columns))
        elif self.data_preproc_hparams.feature_selection_method == "PCA":
            for idx, n in np.ndenumerate(n_features):
                features_support.append(["PCA_"+str(i) for i in np.arange(1,n+1)])
        else:
            raise ValueError("Feature selection method unknown")
        return features_support

    def get_results(self):
        #res = self.results[self.results["inner_split"]==-1].groupby("n_features").mean().reset_index().rename(columns={"acc_train": "acc_train_outer", "acc_val":"acc_test"})
        res_outer_splits_train = self.results[self.results["inner_split"]==-1].groupby(["n_features"])["pred_right_train"].apply(np.hstack).to_frame().reset_index()
        res_outer_splits_test = self.results[self.results["inner_split"]==-1].groupby(["n_features"])["pred_right_val"].apply(np.hstack).to_frame().reset_index()

        res_outer_splits_train["train_acc_mean"] = [pred_right_train.mean() for pred_right_train in res_outer_splits_train["pred_right_train"]]
        res_outer_splits_test["val_acc_mean"] = [pred_right_test.mean() for pred_right_test in res_outer_splits_test["pred_right_val"]]
        res_outer_splits_test = res_outer_splits_test.rename(columns={"pred_right_val": "pred_right_test", "val_acc_mean": "acc_test"})  # rename since for splits -1 val data is the test data

        res_outer_splits = res_outer_splits_train.merge(res_outer_splits_test, on="n_features")

        # gather additional information for the splits
        hparams_names = ["feature_selection_initial", "quantile_transform", "scaler", "feature_selection_method", "n_features"]
        hparams = pd.DataFrame(columns=[*hparams_names, "classifier"])
        acc_val = pd.DataFrame(columns=["n_features", "acc_val_inner"])
        for n_features in self.results["n_features"].unique():
            # calculate mean validation accuracy on the inner splits for n_features. Done by getting pred_right_val for used inner splits 
            pred_right = []
            for split in self.results["outer_split"].unique():
                # get hparams used for outer split (to find all inner splits with this hparam combination)
                hparams_split = self.results[(self.results["n_features"]==n_features) & (self.results["outer_split"]==split) & (self.results["inner_split"]==-1)]["class_hparams"].values
                # get the inner splits with the hparams used by the outer split ("inner_split"!=-1 ) and stack pred_right_val to calculate accuracy later
                pred_right_val_inner = self.results[(self.results["n_features"]==n_features) & (self.results["outer_split"]==split) & (self.results["inner_split"]!=-1) & (self.results["class_hparams"]==hparams_split[0])]
                pred_right_val_inner = pred_right_val_inner.groupby(["outer_split"])["pred_right_val"].apply(np.hstack).iloc[0]

                pred_right = pred_right + list(pred_right_val_inner)
            
            acc_val = acc_val.append(pd.DataFrame([[n_features, np.array(pred_right).mean()]], columns=["n_features", "acc_val_inner"]), ignore_index=True)
            hparams_n_features = self.results[(self.results["n_features"]==n_features) & (self.results["inner_split"]==-1)].iloc[0][hparams_names]
            hparams_n_features["classifier"] = self.classifier_name
            hparams = hparams.append(hparams_n_features)
        res_outer_splits = res_outer_splits.merge(acc_val, on="n_features")
        res_outer_splits = res_outer_splits.merge(hparams, on="n_features")
        res_outer_splits = res_outer_splits[[*hparams_names, "classifier", "pred_right_train", "pred_right_test", "train_acc_mean", "acc_val_inner", "acc_test"]]
        return res_outer_splits

    def get_confusion_matrix(self, n_features):
        cm = confusion_matrix(self.pred_test[n_features]["gt"].values, self.pred_test[n_features]["pred"].values, labels=self.data_preproc_hparams.diagnose_selection)
        return cm

    def get_classification_results(self):
        return self.pred_test

    def scale_transform(self, train, val, feature_selection):
        train.reset_index(inplace=True)
        val.reset_index(inplace=True)  # Needed for allocation of transformed data
        if self.data_preproc_hparams.quantile_transform:
            quantile_transformer = QuantileTransformer(output_distribution="uniform", random_state=0, n_quantiles=50) #n_quantiles 50?
            train[feature_selection] = pd.DataFrame(quantile_transformer.fit_transform(train[feature_selection]), columns=feature_selection)
            val[feature_selection] = pd.DataFrame(quantile_transformer.transform(val[feature_selection]), columns=feature_selection)

        # Scale to zero mean, std 1
        if self.data_preproc_hparams.scaler == "std":
            scaler = StandardScaler()
            train[feature_selection] = pd.DataFrame(scaler.fit_transform(train[feature_selection]), columns=feature_selection)
            val[feature_selection] = pd.DataFrame(scaler.transform(val[feature_selection]), columns=feature_selection)
        elif self.data_preproc_hparams.scaler == "none":
            pass
        else:
            raise ValueError("Scaler unknown")

        if self.data_preproc_hparams.feature_selection_method == "PCA":
            pca = PCA(n_components=self.data_preproc_hparams.feature_selection_num_features.max())
            pca.fit(train[feature_selection])
            pca_feature_names = ["PCA_"+str(i+1) for i in range(self.data_preproc_hparams.feature_selection_num_features.max())]
            train = train.join(pd.DataFrame(pca.transform(train[feature_selection]), columns=pca_feature_names))
            val = val.join(pd.DataFrame(pca.transform(val[feature_selection]), columns=pca_feature_names))
        return train, val


def create_classifiers(hparams, n_splits_outer, n_splits_inner):
    classifiers = {}
    svc_params = {"random_state": [0], "kernel": ["rbf", "poly"], "C": [0.5, 1.0, 2.0], "cache_size": [1000]}  # "kernel": ["linear", "poly", "rbf", "sigmoid"] , "C": [0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    classifiers["svc"] = ClassifierHparamOptimizer(SVC, svc_params, hparams, n_splits_outer, n_splits_inner)

    #knn_params = {"n_neighbors": [2, 5, 10], "algorithm": ["ball_tree", "kd_tree"], "leaf_size": [15, 30, 60]}  #[2, 3, 5, 8, 10], ["ball_tree", "kd_tree"] leaf [5, 10, 20, 30, 50]
    #classifiers["knn"] = ClassifierHparamOptimizer(KNeighborsClassifier, knn_params, hparams, n_splits_outer, n_splits_inner)

    #gnb_params = {}
    #classifiers["gnb"] = ClassifierHparamOptimizer(GaussianNB, gnb_params, hparams, n_splits_outer, n_splits_inner)

    #dt_params = {"random_state": [0]}
    #classifiers["dt"] = ClassifierHparamOptimizer(DecisionTreeClassifier, dt_params, hparams, n_splits_outer, n_splits_inner)

    #logreg_params = {"random_state": [0], "C": [0.5, 1.0, 2.0], "max_iter": [1000]}  # "C": [0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0]
    #classifiers["logreg"] = ClassifierHparamOptimizer(LogisticRegression, logreg_params, hparams, n_splits_outer, n_splits_inner)

    #rf_params = {"random_state": [0], "n_estimators": [2, 5, 10, 50, 100, 200]}
    #classifiers["rf"] = ClassifierHparamOptimizer(RandomForestClassifier, rf_params, hparams, n_splits_outer, n_splits_inner)

    #xgb_params = {'booster': ['gbtree'], 'objective': ["reg:squarederror"], 'eta': [0.3, 0.5, 0.8], 'gamma': [0, 0.5, 1.0],
    #              'max_depth': [2, 6], 'lambda': [1, 1.5, 2.0], 'alpha': [0, 0.1, 0.5]}  #, "binary:logistic"
    #classifiers["xgb"] = ClassifierHparamOptimizer(xgb.XGBClassifier, xgb_params, hparams, n_splits_outer, n_splits_inner)

    return classifiers


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
    n_splits_outer = 5
    n_splits_inner = 4
    classifiers = create_classifiers(hparams, n_splits_outer, n_splits_inner)


    measures = data_loader(hparams.features_folder)  # , patients=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20]
    group_data(measures, hparams.diagnose_groups)
    measures, feature_selection_agg = aggregate_to_patient(measures, hparams.feature_selection_initial, hparams.diagnose_selection)

    measures["split"] = ""
  
    # only keep patient, feature selection, diagnose_grouped
    measures = measures[["sample", "diagnose_grouped", "split", *feature_selection_agg]]

    skf_outer = StratifiedKFold(n_splits=n_splits_outer, random_state=0, shuffle=True)  # train/val:80%, test:20%
    for outer_split_idx, (train_val_index, test_index) in enumerate(skf_outer.split(measures[measures["split"] != "add_train"],
                                                                                    measures[measures["split"] != "add_train"]["diagnose_grouped"])):
        print(f"Outer split Nr.: {outer_split_idx}")
        patients_test = measures[measures["split"] != "add_train"].iloc[test_index]["sample"]
        measures.loc[measures["sample"].isin(patients_test), "split"] = "test"
        patients_train_val = measures[measures["split"] != "add_train"].iloc[train_val_index]["sample"].values  # patient ids to list
        diagnose_train_val = measures[measures["split"] != "add_train"].iloc[train_val_index]["diagnose_grouped"].values
        measures.loc[measures["sample"].isin(patients_train_val), "split"] = "train_val"

        skf_inner = StratifiedKFold(n_splits=n_splits_inner, random_state=0, shuffle=True)  # train/val:80%, test:20%,  4fold creates train: 60%, val:20%
        for inner_split_idx, (train_index, val_index) in enumerate(skf_inner.split(patients_train_val, diagnose_train_val)):
            patients_train = patients_train_val[train_index]
            patients_val = patients_train_val[val_index]
            measures.loc[measures["sample"].isin(patients_train), "split"] = "train"
            measures.loc[measures["sample"].isin(patients_val), "split"] = "val"

            # save split to file
            split_filename = "outer_"+str(outer_split_idx)+"_inner_"+str(inner_split_idx)+".json"
            split_path = hparams.features_folder.parent.joinpath("FBC", hparams.results_folder_name, "splits", split_filename)
            save_split_to_file(measures, split_path)

            for classifier in classifiers.values():
                classifier.optimize(measures[measures["split"]=="train"], measures[measures["split"]=="val"], "diagnose_grouped",
                                    feature_selection_agg, outer_split_idx,
                                    inner_split_idx, additional_training_data=measures[measures["split"]=="add_train"])
        
        for classifier in classifiers.values():
            classifier.optimize(measures[measures["split"].isin(["train", "val"])], measures[measures["split"]=="test"], "diagnose_grouped",
                                feature_selection_agg, outer_split_idx,
                                -1, additional_training_data=measures[measures["split"]=="add_train"])


    # save prediction for classifier
    save_folder_class = hparams.features_folder.parent.joinpath("FBC", hparams.results_folder_name, "detailed_results")
    save_folder_class.mkdir(exist_ok=True, parents=True)


    for classifier_key in classifiers:
        class_res = classifiers[classifier_key].get_classification_results()
        for key, res in class_res.items():
            save_path = save_folder_class.joinpath(create_detailed_results_filename(classifiers[classifier_key], key, ".xlsx"))
            res.to_excel(save_path, index=False)
        res = classifiers[classifier_key].get_results()
        append_df_to_excel(hparams.features_folder.parent.joinpath("FBC", hparams.results_folder_name,"results.xlsx"), res, index=False)

        for n_features in classifiers[classifier_key].data_preproc_hparams.feature_selection_num_features:
            cm = classifiers[classifier_key].get_confusion_matrix(n_features)
            save_path_cm = save_folder_class.joinpath(create_detailed_results_filename(classifiers[classifier_key], n_features, ".png"))
            try:
                save_confusion_matrix(cm, hparams.diagnose_selection, str(type(classifiers[classifier_key].Classifier())).split('.')[-1][:-2], save_path_cm)
            except:
                print("Confusin Matrix could not be saved.")

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
    parser.add_argument("--scaler", type=str, default="none", help="Scaler std or none (std: standard scaler)")
    hparams = parser.parse_args()

    hparams.feature_selection_initial = feature_selection_initial

    if hparams.features_folder == "T":
        hparams.features_folder = features_folder_T
        hparams.diagnose_groups = diagnose_groups_T
    elif hparams.features_folder == "N":
        hparams.features_folder = features_folder_N
        hparams.diagnose_groups = diagnose_groups_N

    hparams.feature_selection_num_features = np.arange(1, hparams.feature_selection_num_features+1)


    # remove:
    hparams.quantile_transform = True
    hparams.feature_selection_method = "sfs"

    main(hparams)
