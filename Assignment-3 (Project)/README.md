# Problem Description

helps us about detecting traces of median filtering in digital images, a problem of big importance in forensics given that filtering can be used to hide traces of image tampering such as re-sampling and light direction. To accomplish this objective, we have used a novel approach based on multiple and multi-scale(different region of interest) progressive perturbations on images able to capture different median filtering traces through using image quality metrics. We have used 8 IQM for each time filter is applied. After getting IQM, we are able to get distinct feature space suitable for proper classification regarding whether or not a given image contains signs of filtering. Experiments using a real-world scenario with compressed and uncompressed images show the effectiveness of our method

==================================================================

We have performed 2 experiments on each FPMW, TPMW, TPOW combinations. We have trained our classifier on TPOW as well, which was not mentioned earlier in experiments section in the paper because of it's poor performance as compared to given two menthods.

# Experiment 1: (compressed images)
1) For testing purpose, we have taken 800 compressed images collected with very different resolutions taken from different cameras and smartphones.
2) The blurred images in the above testing dataset were blurred with different median filtering implementations and different median windows sizes.

# Experiment 2: (compressed + uncompressed images)
1) For testing purpose, we have used 1338 uncompressed images from UCID dataset combined with 800 images from Experiment 1.

For each such experiment, we generate confusion matrix and calculate below model performance metrics based on that:
-Accuracy
-Sensitivity
-Specificity
-Precision
-Recall

We also store feature_vector generated into .txt file.

------------------------------------------------------------------

# How to run the project?

1. Dataset- 
		we will need 2 types of dataset as mentioned in the paper:
			1) Median filtered images - images in .tif format which can be added at location (./dataset/dataset_mf)
			2) Pristine/Original images - images in .tif format which can be added at location (./dataset/dataset_pr)

Note:
-class labels will automatically be generated into feature_vector depending upon the folder the image belongs to. i.e. upon running the project, images in dataset_mf folder will get class = 1(Median Filtered) and images in dataset_pr folder will get class = -1(Pristine) which later can be fed to SVM classifier.
-Also conversion of the input images into Grayscale is done before applying different Image Quality Metrics on it. i.e. you can store both RGB and Grayscale images in above two folders without worrying.

2. if want to run in python IDE:

--go to the location of the project and run below command depending upon what experiment you want to perform.
> python FPMW.py
> python TPMW.py
> python TPOW.py


3. if want to run .py in jupyter notebook:

> %run file_name.py
