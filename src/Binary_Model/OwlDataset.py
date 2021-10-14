import numpy as np
import os
import torch
import torch.utils.data
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms


class OwlDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_annotations,transforms=None):
        # Root is the root directory where images/csv files are located
        self.root = root
        # Transforms are any image modifications
        self.transforms = transforms
        # CSV File of the images and labels
        self.imgs_frame = pd.read_csv(data_annotations)
        # Path to the CSV File
        self.path_to_data_annotations = data_annotations

    def __getitem__(self, index):
        # Open Image and Resize
        image = Image.open(os.path.join(self.root, self.imgs_frame.iloc[index,0])).convert("RGB").resize((100,100))
        # Get label from the data frame for this index
        label = self.imgs_frame.iloc[index,1]
        if self.transforms is not None:
            image = self.transforms(image)
        # Find the File Name
        name = self.imgs_frame.iloc[index,0]
        return image, label, name
    
    def __len__(self):
        return len(self.imgs_frame)

