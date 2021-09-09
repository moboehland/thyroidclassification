Scripts to create the results for the Feature-Based Classification.

The conda VE from the segmentation framework has been used for classification. This commands have been used for creation:
conda create --name ttc_segandfbc python=3.7.7
conda activate ttc_segandfbc
conda install pytorch=1.4.0 pytorch-lightning=0.7.6 albumentations torchvision=0.5.0 opencv pillow numpy=1.16.4 scikit-learn=0.20.2 scikit-image=0.16.2 openpyxl pytables=3.6.1 xgboost skrebate mlxtend seaborn scipy=1.4.1 xlrd jobilb
pip install test-tube

To create the results follow the instructions:
1) Make sure, that the instance segmentation of the desired dataset is present. If not, create the instance segmentation first (see segmentation folder).
2) Run extract_features.py for the desired dataset.
3) The classifiers are trained in classification.py. For possible training hyperparameters see the argparse arguments at the end of the file. See multitrain_N_1.sh, multitrain_N_2.sh and multitrain_L_1.sh, multitrain_L_2.sh for the hyperparameter setups used for the paper. The *.sh files can directly be executed from the FBC folder.

