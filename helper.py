import json
import os

from PIL import Image

images_path = "./images"
labels_path = "./labels"
VISIBILITY_THRESHOLD = 40.00

all_images_paths = []
all_labels_paths = []


def extract_tea_boxes():
    apple_cinnamon_tea_images_path = "./images/cropped2/train/AppleCinnamonTeaBox"
    green_tea_images_path = "./images/cropped2/train/GreenTeaBox"
    blueberry_tea_images_path = "./images/cropped2/train/BlueberryTeaBox"
    count = 0
    apple_cinnamon_count = 0
    green_tea_count = 0
    blueberry_count = 0

    for path in os.listdir(images_path):
        if os.path.isfile(images_path + "/" + path):
            all_images_paths.append(images_path + "/" + path)

    for path in os.listdir(labels_path):
        if os.path.isfile(labels_path + "/" + path):
            all_labels_paths.append(labels_path + "/" + path)

    all_images_paths.sort()
    all_labels_paths.sort()

    for image_path, label_path in zip(all_images_paths, all_labels_paths):
        with open(label_path) as json_file:
            tea_boxes = json.load(json_file)['shapes']
            for tea_box in tea_boxes:
                box_label = tea_box['label']
                visibility = float(tea_box['visibility'][:-1])
                if visibility < VISIBILITY_THRESHOLD or (
                        box_label != "AppleCinnamonTeaBox" and box_label != "GreenTeaBox" and box_label != "BlueberryTeaBox"):
                    continue

                crop_range = tea_box['points']
                crop_range = crop_range[0] + crop_range[1]
                im = Image.open(image_path)
                cropped_im = im.crop(crop_range)

                if box_label == "AppleCinnamonTeaBox":
                    cropped_im.save(apple_cinnamon_tea_images_path + "/" + str(count) + ".png")
                    count += 1
                    apple_cinnamon_count += 1
                    if apple_cinnamon_count > 5529:
                        apple_cinnamon_tea_images_path = "./images/cropped2/test/AppleCinnamonTeaBox"
                    if apple_cinnamon_count > 6426:
                        apple_cinnamon_tea_images_path = "./images/cropped2/validation/AppleCinnamonTeaBox"
                if box_label == "GreenTeaBox":
                    cropped_im.save(green_tea_images_path + "/" + str(count) + ".png")
                    count += 1
                    green_tea_count += 1
                    if green_tea_count > 5253:
                        green_tea_images_path = "./images/cropped2/test/GreenTeaBox"
                    if green_tea_count > 6122:
                        green_tea_images_path = "./images/cropped2/validation/GreenTeaBox"
                if box_label == "BlueberryTeaBox":
                    cropped_im.save(blueberry_tea_images_path + "/" + str(count) + ".png")
                    count += 1
                    blueberry_count += 1
                    if blueberry_count > 5623:
                        blueberry_tea_images_path = "./images/cropped2/test/BlueberryTeaBox"
                    if blueberry_count > 6552:
                        blueberry_tea_images_path = "./images/cropped2/validation/BlueberryTeaBox"


extract_tea_boxes()
