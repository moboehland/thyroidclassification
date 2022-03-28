Scripts to create the results for the Feature-Based Classification.

The conda VE from the segmentation framework has been used for classification. Please see the README.md in the segmentation folder on how to create the VE


To create the results follow the instructions:
1) Make sure, that the instance segmentation of the dataset is present. If not, create the instance segmentation first (see segmentation/README.md).
2) Structure in the dataset_folder should be as follows:
dataset_folder
    img*.tif
    ground_truth.xlsx
    results
        segmentation
            img*_border.png
            img*_instance_rgb.png
            img*_instance.png
The structure is automatically created from the segmentation/inference.py script. You only need to rename the EXPERIMENT_NAME to segmentation. See segmentation/README.md for further information.
4) The ground_truth.xlsx file needs to have the columns ["sample", "diagnose", "diagnose_grouped"]. "diagnose_grouped" can be the same as "diagnose", when no grouping is wanted.
3) Set the folder to the Dataset in extract_features.py with dataset_folder
4) Run extract_features.py for the desired dataset. Features are saved to results/features
5) The classifiers are trained in classification.py with the extracted features. For possible training hyperparameters see the argparse arguments. See multitrain_N_1.sh, multitrain_N_2.sh and multitrain_L_1.sh, multitrain_L_2.sh for the hyperparameter setups used for the paper. The *.sh files can directly be executed from the FBC folder.
6) Results are saved to dataset_folder/FBC/results/results_folder_name

