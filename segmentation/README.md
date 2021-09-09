The python environment needed for the scripts in the segmentation folder has been created with the following commands:
conda create --name ttc_segmentation python=3.7
conda install pytorch=1.4.0 pytorch-lightning=0.7.6 albumentations torchvision=0.5.0 opencv pillow numpy scikit-learn openpyxl pytables=3.6.1 xgboost skrebate mlxtend seaborn
pip install test-tube