import json
import pandas as pd
import os
import PIL
import numpy as np

# make each image an instance of the class Image that can contain
# different information on its subimages, etc.
class Image:
    def __init__(self):
        self.file_name = ''
        self.sub_images = []
        self.contains_owl = False
        self.owl_count = 0


def load_data(json_image_file, images_folder_path):
    img_dir = []
    json_detection_data = json.load(json_image_file)

    # iterate through each of  the images in the json file
    for image in json_detection_data['images']:

        img = Image()  # create an instane of the Image class
        img.file_name = image['file'].replace(images_folder_path, '')  # store the file name

        # detection is a list of dictionaries containing information on each sub-image in the image
        for detections in image['detections']:
            img.sub_images.append(detections['bbox'])

        img_dir.append(img)

    return img_dir

def save_results(filtered_dir, img_dir, images_folder_path):
    # check if the filtered_images dir exists
    if os.path.exists(filtered_dir):
        print("\nAdding images to existing directory:", filtered_dir)
    else:
        os.mkdir(filtered_dir)

    # create a subdir in filtered_dir
    folder_path = os.path.basename(os.path.normpath(images_folder_path))
    filtered_sub_dir = os.path.join(filtered_dir, folder_path)
    # create a new folder with the same name as images_folder_path

    # check if the subdirectory exists
    if os.path.exists(filtered_sub_dir):
        print("\nAdding images to existing sudirectory:", filtered_sub_dir)
    else:
        os.mkdir(filtered_sub_dir)

    os.chdir(filtered_sub_dir)

    file_names = []
    owl_predictions = []
    owl_quantity = []

    print("Saving results to:", os.path.join(filtered_sub_dir, (folder_path + '_owl_predictions.csv\n')))
    # is it possible to just move them to this folder?
    for img in img_dir:
        file_names.append(img.file_name)
        owl_predictions.append(img.contains_owl)
        owl_quantity.append(img.owl_count)
        if img.contains_owl:
            temp_image = PIL.Image.open(images_folder_path+ '/' + img.file_name)
            temp_image.save(img.file_name)
            temp_image.close()

    user_images_predictions = pd.DataFrame(zip(file_names, owl_predictions, owl_quantity), columns=['Image', 'Owl_Prediction', 'Min_Owls_Detected'])
    user_images_predictions.to_csv(folder_path + '_owl_predictions.csv', index=False)

