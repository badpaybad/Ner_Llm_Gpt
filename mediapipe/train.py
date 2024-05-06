# pip install mediapipe-model-maker
# pip install PyQt5
# https://storage.googleapis.com/mediapipe-tasks/object_detector/android_figurine.zip
# https://github.com/google-edge-ai/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb
# sudo apt-get install python3-tk
# pip install ipympl
import math
from collections import defaultdict
from matplotlib import patches, text, patheffects
import matplotlib.pyplot as plt
import matplotlib
print(matplotlib.matplotlib_fname())
import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)
guibackend="Qt5Agg"
matplotlib.use(guibackend)
plt.switch_backend(guibackend)
# # plt.style.use('ggplot')

matplotlib.rcParams['backend'] = guibackend
print(f"Interactive mode: {matplotlib.is_interactive()}")
print(f"matplotlib backend: {matplotlib.rcParams['backend']}")

import numpy as np

# # Generate a random image
# image = np.random.rand(100, 100)
# # Display the image interactively
# plt.imshow(image, cmap='gray')
# plt.colorbar()  # Add a colorbar for better visualization
# plt.show()

# pip install -U jaxlib==0.4.12
from mediapipe_model_maker import object_detector
import os
import json
import tensorflow as tf
assert tf.__version__.startswith('2')

train_dataset_path = "/work/llm/Ner_Llm_Gpt/mediapipe/hannomdataset/train"
validation_dataset_path="/work/llm/Ner_Llm_Gpt/mediapipe/hannomdataset/valid"

with open(os.path.join(train_dataset_path, "labels.json"), "r") as f:
    labels_json = json.load(f)
for category_item in labels_json["categories"]:
    print(f"{category_item['id']}: {category_item['name']}")


def draw_outline(obj):
    obj.set_path_effects(
        [patheffects.Stroke(linewidth=4,  foreground='black'), patheffects.Normal()])


def draw_box(ax, bb):
    patch = ax.add_patch(patches.Rectangle(
        (bb[0], bb[1]), bb[2], bb[3], fill=False, edgecolor='red', lw=2))
    draw_outline(patch)


def draw_text(ax, bb, txt, disp):
    text = ax.text(bb[0], (bb[1]-disp), txt, verticalalignment='top',
                color='white', fontsize=10, weight='bold')
    draw_outline(text)


def draw_bbox(ax, annotations_list, id_to_label, image_shape):
    for annotation in annotations_list:
        cat_id = annotation["category_id"]
        bbox = annotation["bbox"]
        draw_box(ax, bbox)
        draw_text(ax, bbox, id_to_label[cat_id], image_shape[0] * 0.05)


def visualize(dataset_folder, max_examples=None):    
    
    guibackend="Qt5Agg"
    matplotlib.use(guibackend)
    plt.switch_backend(guibackend)
    
    with open(os.path.join(dataset_folder, "labels.json"), "r") as f:
        labels_json = json.load(f)
    images = labels_json["images"]
    cat_id_to_label = {item["id"]: item["name"] for item in labels_json["categories"]}
    image_annots = defaultdict(list)
    for annotation_obj in labels_json["annotations"]:
        image_id = annotation_obj["image_id"]
        image_annots[image_id].append(annotation_obj)

    if max_examples is None:
        max_examples = len(image_annots.items())
    n_rows = math.ceil(max_examples / 3)
    # 3 columns(2nd index), 8x8 for each image
    fig, axs = plt.subplots(n_rows, 3, figsize=(24, n_rows*8))
    for ind, (image_id, annotations_list) in enumerate(list(image_annots.items())[:max_examples]):
        ax = axs[ind//3, ind % 3]
        img = plt.imread(os.path.join(
            dataset_folder, "images", images[image_id]["file_name"]))
        ax.imshow(img)
        draw_bbox(ax, annotations_list, cat_id_to_label, img.shape)       
        # print(ax)
        # plt.imshow(img,cmap="gray")
    
    # plt.colorbar() 
    plt.show()

"""
test GUI backend 
python -c "from PyQt5.QtWidgets import *; app = QApplication([]); win = QMainWindow(); win.show(); app.exec()"
python3 -c "from tkinter import Tk; Tk().mainloop()"

"""

# visualize(train_dataset_path, 9)
train_data = object_detector.Dataset.from_coco_folder(train_dataset_path)#, cache_dir="/tmp/od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path)#, cache_dir="/tmp/od_data/validation")
print("train_data size: ", train_data.size)
print("validation_data size: ", validation_data.size)

spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG
# spec = object_detector.SupportedModels.MOBILENET_V2
#https://developers.google.com/mediapipe/api/solutions/python/mediapipe_model_maker/object_detector/SupportedModels
hparams = object_detector.HParams( 
    learning_rate = 0.3,
    batch_size = 16,
    epochs = 100,
export_dir='exported_model')
options = object_detector.ObjectDetectorOptions(
    supported_model=spec,
    hparams=hparams
)

model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options)

loss, coco_metrics = model.evaluate(validation_data, batch_size=1)
print(f"Validation loss: {loss}")
print(f"Validation coco metrics: {coco_metrics}")

finalmodel = model.export_model()
print(finalmodel)



# qat_hparams = object_detector.QATHParams(learning_rate=0.3, batch_size=4, epochs=10, decay_steps=6, decay_rate=0.96)
# model.quantization_aware_training(train_data, validation_data, qat_hparams=qat_hparams)
# qat_loss, qat_coco_metrics = model.evaluate(validation_data)
# print(f"QAT validation loss: {qat_loss}")
# print(f"QAT validation coco metrics: {qat_coco_metrics}")

# new_qat_hparams = object_detector.QATHParams(learning_rate=0.9, batch_size=4, epochs=15, decay_steps=5, decay_rate=0.96)
# model.restore_float_ckpt()
# model.quantization_aware_training(train_data, validation_data, qat_hparams=new_qat_hparams)
# qat_loss, qat_coco_metrics = model.evaluate(validation_data)
# print(f"QAT validation loss: {qat_loss}")
# print(f"QAT validation coco metrics: {qat_coco_metrics}")

# model.export_model('model_int8_qat.tflite')
