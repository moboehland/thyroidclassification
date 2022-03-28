# thyroidclassification

Code to reproduce the results for "Machine learning methods for automated classification of tumors with papillary thyroid carcinoma-like nuclei: a quantitative analysis".

Feature-based classification results are obtained by training a U-Net for segmentation on the MoNuSeg Challenge data and using it to create an instance segmentation for the desired dataset. Afterwards the instance segmentation together with the thyroid images are used to create features. The features are used to train classifiers.

Deep learning-based classification results are directly obtained by training of classification networks. Since the splits of the feature-based classification are used, it needs to be performed first.

To reproduce the paper results, the following steps need to be performed:
1) Download and preprocess the MoNuSeg Challenge data. See segmentation/prepare_MoNuSeg.py for instructions.
2) Train the segmentation network on the MoNuSeg Challenge data. See segmentation/README.md for instructions.
3) Use the trained model for inference on the Nikiforov and Tharun & Thompson dataset. See segmentation/README.md for instructions.
4) For the Feature-based classification see FBC/README.md for instructions.
5) Perform the deep learning-based classification with the classification_module.py in the DLC folder. See DLC/Readme.md for Instructions