# # pip install mediapipe-model-maker
# # pip install PyQt5
# # https://storage.googleapis.com/mediapipe-tasks/object_detector/android_figurine.zip
# # https://github.com/google-edge-ai/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb
# # sudo apt-get install python3-tk
# # pip install ipympl
# from config import train_dataset_path, validation_dataset_path
# import tensorflow as tf
# import json
# import os
# import mediapipe_model_maker
# from mediapipe_model_maker import object_detector
# import numpy as np
# import matplotlib.rcsetup as rcsetup
# import math
# from collections import defaultdict
# from matplotlib import patches, text, patheffects
# import matplotlib.pyplot as plt
# import matplotlib
# print(matplotlib.matplotlib_fname())
# print(rcsetup.all_backends)
# # guibackend = "Qt5Agg"
# # matplotlib.use(guibackend)
# # plt.switch_backend(guibackend)
# # # plt.style.use('ggplot')

# # matplotlib.rcParams['backend'] = guibackend
# print(f"Interactive mode: {matplotlib.is_interactive()}")
# print(f"matplotlib backend: {matplotlib.rcParams['backend']}")


# # # Generate a random image
# # image = np.random.rand(100, 100)
# # # Display the image interactively
# # plt.imshow(image, cmap='gray')
# # plt.colorbar()  # Add a colorbar for better visualization
# # plt.show()

# # pip install -U jaxlib==0.4.12
# assert tf.__version__.startswith('2')

# # train_dataset_path = "/work/Ner_Llm_Gpt/mediapipe/chunomdataset/train"
# # validation_dataset_path = "/work/Ner_Llm_Gpt/mediapipe/chunomdataset/valid"


# with open(os.path.join(train_dataset_path, "labels.json"), "r") as f:
#     labels_json = json.load(f)
# for category_item in labels_json["categories"]:
#     print(f"{category_item['id']}: {category_item['name']}")


# def draw_outline(obj):
#     obj.set_path_effects(
#         [patheffects.Stroke(linewidth=4,  foreground='black'), patheffects.Normal()])


# def draw_box(ax, bb):
#     patch = ax.add_patch(patches.Rectangle(
#         (bb[0], bb[1]), bb[2], bb[3], fill=False, edgecolor='red', lw=2))
#     draw_outline(patch)


# def draw_text(ax, bb, txt, disp):
#     text = ax.text(bb[0], (bb[1]-disp), txt, verticalalignment='top',
#                    color='white', fontsize=10, weight='bold')
#     draw_outline(text)


# def draw_bbox(ax, annotations_list, id_to_label, image_shape):
#     for annotation in annotations_list:
#         cat_id = annotation["category_id"]
#         bbox = annotation["bbox"]
#         draw_box(ax, bbox)
#         draw_text(ax, bbox, id_to_label[cat_id], image_shape[0] * 0.05)


# def visualize(dataset_folder, max_examples=None):

#     guibackend = "Qt5Agg"
#     matplotlib.use(guibackend)
#     plt.switch_backend(guibackend)

#     with open(os.path.join(dataset_folder, "labels.json"), "r") as f:
#         labels_json = json.load(f)
#     images = labels_json["images"]
#     cat_id_to_label = {item["id"]: item["name"]
#                        for item in labels_json["categories"]}
#     image_annots = defaultdict(list)
#     for annotation_obj in labels_json["annotations"]:
#         image_id = annotation_obj["image_id"]
#         image_annots[image_id].append(annotation_obj)

#     if max_examples is None:
#         max_examples = len(image_annots.items())
#     n_rows = math.ceil(max_examples / 3)
#     # 3 columns(2nd index), 8x8 for each image
#     fig, axs = plt.subplots(n_rows, 3, figsize=(24, n_rows*8))
#     for ind, (image_id, annotations_list) in enumerate(list(image_annots.items())[:max_examples]):
#         ax = axs[ind//3, ind % 3]
#         img = plt.imread(os.path.join(
#             dataset_folder, "images", images[image_id]["file_name"]))
#         ax.imshow(img)
#         draw_bbox(ax, annotations_list, cat_id_to_label, img.shape)
#         # print(ax)
#         # plt.imshow(img,cmap="gray")

#     # plt.colorbar()
#     plt.show()


# """
# test GUI backend 
# python -c "from PyQt5.QtWidgets import *; app = QApplication([]); win = QMainWindow(); win.show(); app.exec()"
# python3 -c "from tkinter import Tk; Tk().mainloop()"

# """

# # visualize(train_dataset_path, 9)
# train_data = object_detector.Dataset.from_coco_folder(
#     train_dataset_path)  # , cache_dir="/tmp/od_data/train")
# validation_data = object_detector.Dataset.from_coco_folder(
#     validation_dataset_path)  # , cache_dir="/tmp/od_data/validation")
# print("train_data size: ", train_data.size)
# print("validation_data size: ", validation_data.size)

# # 2. Cấu hình ban đầu
# spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG
# total_epochs = 1
# step_size = 1000
# export_main_dir = 'exported_model'
# # Biến để lưu trữ lịch sử loss qua các vòng lặp
# history_all = {'loss': [], 'val_loss': []}

# # 2. Load dữ liệu
# train_data = object_detector.Dataset.from_coco_folder(train_dataset_path)
# validation_data = object_detector.Dataset.from_coco_folder(
#     validation_dataset_path)

# spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG

# # 3. Vòng lặp huấn luyện
# model = None

# # hparams = object_detector.HParams(
# #     learning_rate=0.1,
# #     batch_size=16,
# #     epochs=1 ,
# #     export_dir=export_main_dir
# # )

# # options = object_detector.ObjectDetectorOptions(
# #     supported_model=spec,
# #     hparams=hparams
# # )

# # model = object_detector.ObjectDetector.create(
# #     train_data=train_data,
# #     validation_data=validation_data,
# #     options=options,
# # )
    
# for epoch_checkpoint in range(step_size, total_epochs + step_size, step_size):
#     print(f"\n--- GIAI ĐOẠN: {epoch_checkpoint} EPOCHS ---")

#     hparams = object_detector.HParams(
#         learning_rate=0.1,
#         batch_size=16,
#         epochs=1,
#         export_dir=export_main_dir
#     )

#     options = object_detector.ObjectDetectorOptions(
#         supported_model=spec,
#         hparams=hparams
#     )
#     if model is None:
#         model = object_detector.ObjectDetector.create(
#             train_data=train_data,
#             validation_data=validation_data,
#             options=options,
#         )
#     else:
#         model = tf.keras.models.load_model(f'my_saved_model_folder_{epoch_checkpoint-1}')
#         # 2. (Tùy chọn) Compile lại để chỉnh Learning Rate
#         # MediaPipe Object Detector thường dùng Adam hoặc SGD
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), # Hạ thấp LR
#             loss=None # Loss đã được định nghĩa trong kiến trúc model khi save
#         )
#         # 3. Gọi hàm fit để train tiếp
#         model.fit(
#             train_data,
#             validation_data=validation_data,
#             epochs=total_epochs # Huấn luyện thêm 1000 epoch nữa
#         )

#     # Lưu lịch sử loss
#     # model.model là đối tượng Keras thuần túy bên dưới MediaPipe
#     history_all['loss'].extend(model._model.history.history['loss'])
#     history_all['val_loss'].extend(model._model.history.history['val_loss'])

#     # --- VẼ BIỂU ĐỒ ---
#     plt.figure(figsize=(10, 5))
#     plt.plot(history_all['loss'], label='Train Loss')
#     plt.plot(history_all['val_loss'], label='Validation Loss')
#     plt.title(f'Model Loss up to {epoch_checkpoint} Epochs')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend()
#     plt.grid(True)

#     # Lưu biểu đồ thành file ảnh
#     plt.savefig(f'loss_chart_{epoch_checkpoint}.png')
#     plt.close()  # Giải phóng bộ nhớ RAM sau khi vẽ

#     # Lưu model
#     model.export_model(f'model_epochf_{epoch_checkpoint}.tflite')
#     print(f"Đã lưu model và biểu đồ tại mốc {epoch_checkpoint}")
#     # Trong Keras (lớp dưới của MediaPipe)
#     model._model.save(f'my_saved_model_folder_{epoch_checkpoint}')

#     model._model.save_weights(f'checkpoint_epoch_{epoch_checkpoint}.weights.h5')

#     # # Khi muốn dùng lại để train tiếp:
#     # import tensorflow as tf
#     # new_model = tf.keras.models.load_model('my_saved_model_folder')

# # 4. Đánh giá cuối cùng
# # loss, coco_metrics = model.evaluate(validation_data, batch_size=1)
# # print(f"Final Loss: {loss}")

# loss, coco_metrics = model.evaluate(validation_data, batch_size=1)
# print(f"Validation loss: {loss}")
# print(f"Validation coco metrics: {coco_metrics}")

# finalmodel = model.export_model()
# print(finalmodel)

# """
# . Phân tích kết quả Training của bạn

# Nhìn vào log dòng cuối cùng: 4/4 [==============================] - 18s 2s/step - total_loss: 6.0702 - cls_loss: 4.6368 - box_loss: 0.0274 ...

#     total_loss (6.0702): Khá cao. Thường thì khi model bắt đầu hội tụ tốt, con số này nên giảm xuống dưới 1.0 hoặc thấp hơn tùy vào dữ liệu.

#     cls_loss (4.6368): Đây là loss về phân loại nhãn. Con số này cao cho thấy model đang gặp khó khăn trong việc nhận diện "đây là vật gì".

#     box_loss (0.0274): Rất thấp và tốt. Điều này có nghĩa là model xác định vị trí khung hình (Bounding Box) khá ổn, nhưng chưa biết gọi tên vật đó là gì.
# Lời khuyên để cải thiện

# Vì bạn đang train 10.000 epoch (một con số rất lớn), hãy chú ý:

#     Theo dõi val_cls_loss: Nếu sau khoảng 2000-3000 epoch mà val_cls_loss không giảm thêm hoặc bắt đầu tăng lại, đó là dấu hiệu của Overfitting (học vẹt). Bạn nên dừng lại ở mốc đó thay vì chạy hết 10.000.

#     Dữ liệu: Với batch_size=16 và chỉ có 4/4 step mỗi epoch, tổng dữ liệu của bạn có vẻ hơi ít (khoảng 64 ảnh). Với tập dữ liệu nhỏ, việc train 10.000 epoch chắc chắn sẽ gây Overfitting.

#     Hạ Learning Rate: Nếu sau khi train một thời gian mà loss vẫn dao động quanh mức 6.0 không giảm, hãy thử hạ learning_rate=0.1 xuống 0.01 hoặc 0.001.
# """

# """for cpu"""
# # qat_hparams = object_detector.QATHParams(learning_rate=0.3, batch_size=4, epochs=10, decay_steps=6, decay_rate=0.96)
# # model.quantization_aware_training(train_data, validation_data, qat_hparams=qat_hparams)
# # qat_loss, qat_coco_metrics = model.evaluate(validation_data)
# # print(f"QAT validation loss: {qat_loss}")
# # print(f"QAT validation coco metrics: {qat_coco_metrics}")

# # new_qat_hparams = object_detector.QATHParams(learning_rate=0.9, batch_size=4, epochs=15, decay_steps=5, decay_rate=0.96)
# # model.restore_float_ckpt()
# # model.quantization_aware_training(train_data, validation_data, qat_hparams=new_qat_hparams)
# # qat_loss, qat_coco_metrics = model.evaluate(validation_data)
# # print(f"QAT validation loss: {qat_loss}")
# # print(f"QAT validation coco metrics: {qat_coco_metrics}")

# # model.export_model('model_int8_qat.tflite')

# """
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt update

# sudo apt install software-properties-common

# sudo apt install python3.9 python3.9-venv python3.9-dev

# rm -rf venv_mediapipe
# python3.9 -m venv venv_mediapipe
# source venv_mediapipe/bin/activate


# git clone https://github.com/google/mediapipe.git
# cd mediapipe/mediapipe/model_maker

# pip install -r requirements.txt
# pip install .

# """


# def inference():

#     import mediapipe as mptask
#     from mediapipe.tasks.python import vision

#     # 1. Cấu hình các tùy chọn cho Detector
#     model_path = 'model_epoch_1000.tflite'  # Đường dẫn đến file bạn đã save

#     options = vision.ObjectDetectorOptions(
#         base_options=mptask.BaseOptions(model_asset_path=model_path),
#         score_threshold=0.5,  # Chỉ lấy các đối tượng có độ tự tin > 50%
#         running_mode=vision.RunningMode.IMAGE
#     )

#     # 2. Khởi tạo detector
#     with vision.ObjectDetector.create_from_options(options) as detector:

#         # 3. Load ảnh (Sử dụng mediapipe.Image)
#         image = mptask.Image.create_from_file("path/to/your/image.jpg")

#         # 4. Chạy inference
#         detection_result = detector.detect(image)

#         # 5. Xử lý kết quả
#         for detection in detection_result.detections:
#             category = detection.categories[0]
#             print(f"Nhãn: {category.category_name}")
#             print(f"Độ tự tin: {category.score:.2f}")
#             print(f"Tọa độ (Bounding Box): {detection.bounding_box}")


# def embeding():
#     import mediapipe as mp
#     from mediapipe.tasks import python
#     from mediapipe.tasks.python import vision

#     # Cấu hình Embedder
#     base_options = python.BaseOptions(
#         model_asset_path='model_epoch_1000.tflite')
#     options = vision.ImageEmbedderOptions(base_options=base_options)

#     with vision.ImageEmbedder.create_from_options(options) as embedder:
#         # Chuyển đổi ảnh
#         image = mp.Image.create_from_file("test_image.jpg")

#         # Lấy embedding
#         embedding_result = embedder.embed(image)

#         # Vector đặc trưng nằm ở đây
#         vector = embedding_result.embeddings[0].float_embedding
#         print(f"Độ dài vector: {len(vector)}")
#         print(f"Vector: {vector[:5]}...")  # Xem 5 giá trị đầu


# def detect_embeding():
#     import cv2
#     import numpy as np
#     import mediapipe as mp
#     from mediapipe.tasks import python
#     from mediapipe.tasks.python import vision

#     # https://ai.google.dev/edge/mediapipe/solutions/vision/image_embedder
#     # https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/latest/mobilenet_v3_large.tflite
#     # --- 1. Cấu hình đường dẫn model ---
#     detector_path = 'model_epoch_1000.tflite'
#     # Bạn có thể dùng model mặc định của MediaPipe
#     embedder_path = 'your_embedder_model.tflite'

#     # --- 2. Khởi tạo Detector và Embedder ---
#     base_options_det = python.BaseOptions(model_asset_path=detector_path)
#     options_det = vision.ObjectDetectorOptions(
#         base_options=base_options_det, score_threshold=0.5)
#     detector = vision.ObjectDetector.create_from_options(options_det)

#     base_options_emb = python.BaseOptions(model_asset_path=embedder_path)
#     options_emb = vision.ImageEmbedderOptions(base_options=base_options_emb)
#     embedder = vision.ImageEmbedder.create_from_options(options_emb)

#     def get_object_embeddings(image_path):
#         # Đọc ảnh bằng OpenCV để dễ dàng cắt (crop)
#         cv_image = cv2.imread(image_path)
#         rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

#         # BƯỚC 1: Detect đối tượng
#         detection_result = detector.detect(mp_image)

#         results = []

#         for detection in detection_result.detections:
#             bbox = detection.bounding_box
#             x, y, w, h = int(bbox.origin_x), int(
#                 bbox.origin_y), int(bbox.width), int(bbox.height)

#             # Cắt vùng ảnh chứa đối tượng (Crop)
#             # Đảm bảo tọa độ không nằm ngoài biên ảnh
#             x, y = max(0, x), max(0, y)
#             cropped_img = rgb_image[y:y+h, x:x+w]

#             if cropped_img.size == 0:
#                 continue

#             # Chuyển vùng đã cắt sang định dạng MediaPipe Image
#             mp_cropped_img = mp.Image(
#                 image_format=mp.ImageFormat.SRGB, data=cropped_img)

#             # BƯỚC 2: Lấy Embedding cho vùng ảnh vừa cắt
#             embedding_result = embedder.embed(mp_cropped_img)
#             vector = embedding_result.embeddings[0].float_embedding

#             results.append({
#                 "label": detection.categories[0].category_name,
#                 "bbox": [x, y, w, h],
#                 "embedding": vector
#             })

#         return results

#     # --- 3. Chạy thử ---
#     data = get_object_embeddings("test.jpg")
#     for item in data:
#         print(
#             f"Đối tượng: {item['label']} - Kích thước vector: {len(item['embedding'])}")
