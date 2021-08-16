# Owl-Classifier

## Table of contents

1. [Overview](#Owl-Classifier-Overview)
2. [Setup](#Setup-for-OwlDetect)
3. [How to Run](#How-to-run-OwlDetect)
4. [Notes](#Notes)

## Owl Classifier Overview
   - We are currently working on:
     - Adjusting our model to identify other animals in the non-Owl category
   
## Setup for OwlDetect
1. Navigate to the Owl_Classifier via termial 
   - Guide to navigate your machine via terminal: https://openclassrooms.com/en/courses/4614926-learn-the-command-line-in-terminal/4634356-navigate-your-system
   - Download python packages using  *pip install -r requirements.txt* 

2. Download all the necessary files from the links provided
   - Download [Frozen model (.pb)]https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb) and save it in the folder **'~/Owl_Classifier/MegaDetector/cameratraps/detection'**
   - Download [PyTorch_Binary_Classifier.pth]https://drive.google.com/file/d/1y1A23D9wYJBEJ9XTv1Gd8qPiC8SOwVSM/view?usp=sharing) and save it in the folder **'~/Owl_Classifier/Binary_Model'**

4. Once this is done no other steps are needed for the setup. **Remember the location of the file 'Owl_model'**
Note: Setup procedures only needs to be ran once on your machine to run 'terminal_interface.py'


## How to run OwlDetect
1. Get the directory of images you would like analyzed prepared for the pipeline. **The contents of this directory should only be of just images. It will take a while so use a small set if you don't want to wait that long. Unexpected errors may occur if non image files are in directory.**
2. Run terminal_interface.py
   - In the terminal, run *python3 terminal_interface.py*
   - First prompt will ask for the directory path of the directory of images you would like to have analyzed.
   - The MegaDetector will run at this point and this may take some time. **Nothing needs to be done in this step.**
3. The output of this script will be one CSV file and a folder containing images containing owls:
   - ~/Owl_Classifier/filtered_images/YOUR_IMAGE_FILE_PATH/YOUR_IMAGE_FILE_PATH_owl_predictions.csv has the models predictions for the presence of an owl for each image, the file name, and the minimum number of owls
   - ~/Owl_Classifier/filtered_images/YOUR_IMAGE_FILE_PATH contains all images in the give folder that was predicted the presence of an owl

## Notes
   - If you don't rename the files the next time this pipeline is ran these CSV files will be overwritten


## Other folders in Owl-Classifier


### MegaDetector
- [Link for MegaDetector](https://github.com/microsoft/CameraTraps)


### Tools
- **"run_owl_model.py"**: This script runs the owl classifier. Accepts an array of all *Image* object and iterates through that to make predictions 
- **"Image.py"**: This script creates the *Image* object that contains information the image file name, if an owl is contained in an image, the coordinate data on the bounding boxes in an image, and the count of owls. It also contains the function to save the results. 
- **"image_processing.py"**: This script creates the functions to denormalize the bounding boxes coordinates and a function to perform nonmaximum suppression, 