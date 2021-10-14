import PIL
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

def run_model(MODEL_NAME, img_dir, images_folder_path):
    output_dimensions = 1

    # Transformation to image
    # Need ToTensor and Normalize
    transform = transforms.Compose(
        [
            transforms.CenterCrop(100),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
       )

    # RESNET 18
    # However, torchvision has other models we can experiment with if needed
    print("Creating baseline model architecture")
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        torch.nn.Linear(512, output_dimensions),
        torch.nn.Sigmoid()
    )
    print("SUCCESS: Creating baseline model architecture")

    print("Loading in Model: ", MODEL_NAME)
    if(torch.cuda.is_available()):
        model.load_state_dict(torch.load(MODEL_NAME))
        print("SUCCESS: Loading in Model: ", MODEL_NAME)
        print("Using Cuda")
    else:
        model.load_state_dict(torch.load(MODEL_NAME, map_location="cpu"))
        print("SUCCESS: Loading in Model: ", MODEL_NAME)
        print("Using CPU")

    # If cuda is available, will use that or then your computer's CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device Intialized")

    # validate the model
    model.eval()

    RedLineLength = 3

    # iterate through each image object in the list
    for img in img_dir:
        # load the main image and resize
        main_img = PIL.Image.open(images_folder_path + img.file_name).convert("RGB")
        #main_img.show()

        for sub_img in img.sub_images:
            # left,top,right,bottom
            sub = main_img.crop((sub_img[0] + RedLineLength, sub_img[1] + RedLineLength,
                            sub_img[2] - RedLineLength, sub_img[3] - RedLineLength)).resize((100,100))

            #sub.show()
            sub = transform(sub)

            if torch.cuda.is_available():
                sub = sub.cuda()
            output = model(sub[None, ...])
            # Threshold
            rslt = (output) > 0.5
            # if the prediction is true, change the value of img.contains_owl to true
            if (rslt[0][0] == True):
                img.contains_owl = True
                img.owl_count += 1

        main_img.close()

    return img_dir