# used to get the file names in a folder or directory
import os
import subprocess
import json
import Image
from Image import save_results
from run_owl_model import run_model
from image_processing import denormalize
from image_processing import non_max_sup

main_working_dir=os.path.abspath(os.curdir)#current working Directory

detector_working_dir=os.path.join(main_working_dir,'src/MegaDetector/cameratraps/detection/')
print(os.path.join(main_working_dir,"src/MegaDetector/cameratraps/detection/"))
print(detector_working_dir)

# set the version of megatector
param_detector_version="md_v4.1.0.pb"
param_threshold="0.1"

# get the file path containing the images from the user
images_folder_path = input('Enter the file path of images (Ex. ~/Users/admin/Desktop/owl_images): ')

if images_folder_path[-1] != '/':
    images_folder_path = images_folder_path + '/'

# change to hardcode directory path for images
#images_folder_path = ''

# check if the file exists on the machine
if not os.path.exists(images_folder_path):
        raise NotADirectoryError("No such file or directory")
else:
	print('\nLocating Directory...Found')

print('Processing Images...\n')

# just create a temporary json file
json_file_name = os.path.join(main_working_dir, "detectorjsonFile.json")

with open(json_file_name,'w') as outfile :
    pass #this creates the file and just closes it

json_file_path=os.path.join(main_working_dir,json_file_name)

# change the directory to start working with the mega detector files
os.chdir(detector_working_dir)

# ask at next meeting if the system should only support python 3
cmd= "python3 run_tf_detector_batch.py "+ param_detector_version  + " " + images_folder_path + " " + json_file_path + " --recursive"
os.system(cmd)

os.chdir(main_working_dir)

#open and load the json file containing output from the megadetector model
json_image_file = open(json_file_path, )

# will contain all of the images in a list of Image objects
img_dir = Image.load_data(json_image_file,images_folder_path)

# close the json file
json_image_file.close()

# denormalize the image coordinates
img_dir = denormalize(img_dir, images_folder_path)

# implement non max suppression
img_dir = non_max_sup(img_dir, 0.6)

MODEL_PATH = os.path.join(main_working_dir,'Binary_Model',"PyTorch_Binary_Classifier.pth")
img_dir = run_model(MODEL_PATH, img_dir, images_folder_path)

# create the output as a csv file_name, owl
filtered_dir = os.path.join(main_working_dir, 'filtered_images')

save_results(filtered_dir, img_dir, images_folder_path)