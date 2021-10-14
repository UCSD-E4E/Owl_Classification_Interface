import PIL
import numpy as np

# accepts an Image() object
def denormalize(img_dir, images_folder_path):
    main_img = PIL.Image.open(images_folder_path + img_dir[0].file_name).convert("RGB")
    width, height = main_img.size
    main_img.close()

    for img in img_dir:
        for sub_img in img.sub_images:
            one = sub_img[0] * width  # xmin
            two = sub_img[1] * height  # ymin
            three = sub_img[2] * width + sub_img[0] * width  # box width
            four = sub_img[3] * height + sub_img[1] * height  # box height

            sub_img[0] = one
            sub_img[1] = two
            sub_img[2] = three
            sub_img[3] = four

    return img_dir


def non_max_sup(img_dir, areaFactor):

    for img in img_dir:
        if len(img.sub_images) == 0:
            continue

        # convert the subimage to np.array
        boxes = np.array(img.sub_images)
        boxes = boxes.astype("float")
        pick = []

        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            maxX1 = np.maximum(x1[i], x1[idxs[:last]])
            maxY1 = np.maximum(y1[i], y1[idxs[:last]])
            maxX2 = np.minimum(x2[i], x2[idxs[:last]])
            maxY2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, maxX2 - maxX1 + 1)
            h = np.maximum(0, maxY2 - maxY1 + 1)
            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > areaFactor)[0])))

        img.sub_images = boxes[pick].astype("int")

    return img_dir