# Train and evaluate all hyperparameter combinations where all classifiers (svc, knn, gnb, dt, logreg, rf) can be applied. Make sure, that svc is not disabled in the classification.py script.
# To do so, check that line two and three are not commented out in create_classifiers in classification.py


python classification.py --feature_selection_num_features 25 --feature_selection_method chi2 --results_folder_name paper_py37_pl_negpos --scaler std --quantile_transform --features_folder N
python classification.py --feature_selection_num_features 25 --feature_selection_method chi2 --results_folder_name paper_py37_pl_negpos --scaler none --quantile_transform --features_folder N
python classification.py --feature_selection_num_features 25 --feature_selection_method chi2 --results_folder_name paper_py37_pl_negpos --scaler std --features_folder N

python classification.py --feature_selection_num_features 25 --feature_selection_method sfs --results_folder_name paper_py37_pl_negpos --scaler none --quantile_transform --features_folder N
python classification.py --feature_selection_num_features 25 --feature_selection_method sfs --results_folder_name paper_py37_pl_negpos --scaler std --features_folder N
python classification.py --feature_selection_num_features 25 --feature_selection_method sfs --results_folder_name paper_py37_pl_negpos --scaler std --quantile_transform --features_folder N

python classification.py --feature_selection_num_features 25 --feature_selection_method none --results_folder_name paper_py37_pl_negpos --scaler none --quantile_transform --features_folder N
python classification.py --feature_selection_num_features 25 --feature_selection_method none --results_folder_name paper_py37_pl_negpos --scaler std --features_folder N
python classification.py --feature_selection_num_features 25 --feature_selection_method none --results_folder_name paper_py37_pl_negpos --scaler std --quantile_transform --features_folder N
 
