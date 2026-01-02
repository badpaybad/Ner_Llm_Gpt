
import sys
import os
import cv2
import shutil
import json
# sourceChuNomFolder = "/work/Ner_Llm_Gpt/mediapipe/train-val"
# cocoChuNomFolder = "/work/Ner_Llm_Gpt/mediapipe/chunomdataset"
# cocoChuNomFolderTrain = "/work/Ner_Llm_Gpt/mediapipe/chunomdataset/train"
# cocoChuNomFolderValid = "/work/Ner_Llm_Gpt/mediapipe/chunomdataset/valid"
# cocoChuNomFolderTrainImages = "/work/Ner_Llm_Gpt/mediapipe/chunomdataset/train/images"
# cocoChuNomFolderValidImages = "/work/Ner_Llm_Gpt/mediapipe/chunomdataset/valid/images"

from config import sourceChuNomFolder,cocoChuNomFolder,cocoChuNomFolderTrain,cocoChuNomFolderValid,cocoChuNomFolderTrainImages,cocoChuNomFolderValidImages

# class x center y center width height
# ./wb_localization_dataset/labels/val/nlvnpf-0137-01-045.txt
#               0 0.882222 0.871589 0.037778 0.041734
# ./wb_localization_dataset/images/val/nlvnpf-0137-01-045.jpg

# sudo apt install libgtk2.0-dev pkg-config
# pip3 install mediapipe mediapipe_model_maker
# # pip3 install opencv-python opencv-contrib-python


def copy_files(source_dir, destination_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Iterate through files in source directory
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        # Copy the file
        shutil.copy2(source_file, destination_file)
        print(f"File '{filename}' copied to '{destination_dir}'.")


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")


create_directory_if_not_exists(cocoChuNomFolder)
create_directory_if_not_exists(cocoChuNomFolderTrain)
create_directory_if_not_exists(cocoChuNomFolderValid)
create_directory_if_not_exists(cocoChuNomFolderTrainImages)
create_directory_if_not_exists(cocoChuNomFolderValidImages)

copy_files(f"{sourceChuNomFolder}/wb_localization_dataset/images/train",
           cocoChuNomFolderTrainImages)
copy_files(f"{sourceChuNomFolder}/wb_localization_dataset/images/val",
           cocoChuNomFolderValidImages)


def get_file_name_without_extension(file_path):
    # Get the base name of the file (without directory)
    file_name = os.path.basename(file_path)
    # Split the file name into name and extension
    name, extension = os.path.splitext(file_name)
    return name


def center_to_corner(x, y, w, h, imgW, imgH):
    return int((x-w/2)*imgW), int((y-h/2)*imgH), int((x+w/2)*imgW), int((y+h/2)*imgH)


def convert(dir_labels_train=f"{sourceChuNomFolder}/wb_localization_dataset/labels/train", cocoDirImages=cocoChuNomFolderTrainImages, cocoJsonFile=cocoChuNomFolderTrain):

    catid = 1
    images = []
    categories = []
    annotations = []
    categories.append({"id": catid, "name": "chunom"})

    imageId = 0
    for filename in os.listdir(dir_labels_train):
        # Check if the path is a file
        file_path = os.path.join(dir_labels_train, filename)

        filename0Ext = get_file_name_without_extension(file_path)

        if os.path.isfile(file_path):
            img = cv2.imread(f"{cocoDirImages}/{filename0Ext}.jpg")
            images.append({
                "id": imageId,
                "file_name": f"{filename0Ext}.jpg"
            })
            imgh, imgw, imgc = img.shape

            with open(file_path, 'r') as file:
                # Read the entire contents of the file
                file_contents = file.read()
                lines = file_contents.splitlines()
                for l in lines:
                    l = l.strip().replace("  ", " ")
                    words = l.split()
                    x = float(words[1])
                    y = float(words[2])
                    w = float(words[3])
                    h = float(words[4])

                    x, y, x1, y1 = center_to_corner(x, y, w, h, imgw, imgh)

                    drawed = cv2.rectangle(
                        img, (x, y), (x1, y1), (0, 255, 0), 1)
                    w = x1-x
                    h = y1-y
                    # cv2.imshow("Image with Rectangle", drawed)
                    # cv2.waitKey(0)
                    annotations.append({
                        "image_id": imageId,
                        "bbox": [x, y, w, h],
                        "category_id": catid
                    })

            imageId = imageId+1

    cocodata = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(f"{cocoJsonFile}/labels.json", 'w') as file:
        text = json.dumps(cocodata)
        file.write(text)


convert(f"{sourceChuNomFolder}/wb_localization_dataset/labels/train",
        cocoChuNomFolderTrainImages, cocoChuNomFolderTrain)

convert(f"{sourceChuNomFolder}/wb_localization_dataset/labels/val",
        cocoChuNomFolderValidImages, cocoChuNomFolderValid)
