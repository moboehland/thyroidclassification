Use following commands from the DLC directory to create a conda environment: \
conda env create -f requirements.yml \
conda env activate ttc_dlc

Please note, that the environment is only used for the deep learning-based classification.
A separate environment needs to be created for the segmentation and the feature-based classification.

To create results for a dataset run trainer.py. \
Keep in mind, that splits created by the feature-based classification are needed for training. To create the splits run the feature-based classification first. \
When using the framework for a custom dataset, change of the pytorch dataset (thyroid_dataset.py) may be needed. \
To run the scripts, the --dataset_folder with the images and the ground_truth.xlsx as well as the --splits_folder have to be set. \
For more information on the ground_truth.xlsx file, see FBC/README.md

A new name is needed for every run. Do not start a new training with the same --name. \
Keep in mind, that the final models are saved in the lightning_logs/Dataset/Name/Split/Version folder and not in the dataset folder.

After training, run the trainer.py file with "test" for --mode. The results are saved to lightning_logs/Dataset/Name/