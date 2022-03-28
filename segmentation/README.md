Use the following commands from the segmentation directory to create a conda environment: \
conda env create -f requirements.yml \
conda env activate ttc_seg_fbc

Please note, that the environment is only used for segmentation and feature based classification. \
A separate environment needs to be created for the deep learning classification.

Training: \
Run trainer.py and make sure you downloaded and preprocessed the MoNuSeg Challenge data before. See segmentation/prepare_MoNuSeg.py for instructions. \
For different datasets, you can use the --root_dir argument

Inference: \
- Run inference.py
- Set --root_dir to the direction with the images to predict
- Set --checkpoint_path to the checkpoint file of the trained model. 
- Checkpoints are stored in the folder results/EXPERIMENTNAME_VERSION/*.ckpt
- A greyscale instance segmentation (_instance) a rgb instance segmentation (_instance_rgb) and the border prediction (_border) will be saved to a results folder in the root_dir
- When images in a custom dataset are not present in .tif format, please change the ending in the init of histopathology_dataset.py file accordingly.
- The results will be saved into results/EXPERIMENT_NAME folder created in the rood_dir folder of the dataset. EXPERIMENT_NAME is of the experiment given for training.
- To further process with the feature based classification change the EXPERIMENT_NAME folder of your final model to 'segmentation' afterwards