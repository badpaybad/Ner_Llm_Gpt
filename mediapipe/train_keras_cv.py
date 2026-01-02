# # 1. Gỡ sạch các bản hiện tại để tránh xung đột
# pip uninstall -y tensorflow keras keras-cv tensorflow-estimator tf-keras

# # 2. Cài đặt bộ 3 ổn định (TensorFlow 2.15 + KerasCV tương thích)
# pip install tensorflow==2.15.0 keras-cv==0.8.2
import os
import json
import tensorflow as tf
import keras_cv
import keras
from tensorflow import data as tf_data

# Cấu hình
BATCH_SIZE = 8
IMAGE_SIZE = (640, 640)
NUM_CLASSES = 80 # Số class trong dataset của bạn (COCO mặc định là 80)
AUTOTUNE = tf_data.AUTOTUNE

# Hàm đọc ảnh và load label từ COCO JSON
def load_coco_json(json_path, image_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img['file_name'] for img in data['images']}
    annotations = {}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = {'boxes': [], 'classes': []}
        # COCO format: [x_min, y_min, width, height]
        # KerasCV format: [x_min, y_min, width, height] (xywh) hoặc format khác tùy config
        annotations[img_id]['boxes'].append(ann['bbox'])
        annotations[img_id]['classes'].append(ann['category_id'])
        
    return images, annotations

# Tạo tf.data.Dataset Generator
def make_dataset(image_dir, annotation_file):
    images_map, ann_map = load_coco_json(annotation_file, image_dir)
    image_ids = list(images_map.keys())

    def generator():
        for img_id in image_ids:
            img_path = os.path.join(image_dir, images_map[img_id])
            if img_id not in ann_map: continue # Bỏ qua ảnh không có box
            
            # Load ảnh
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMAGE_SIZE)
            
            # Load box & label
            boxes = tf.constant(ann_map[img_id]['boxes'], dtype=tf.float32)
            classes = tf.constant(ann_map[img_id]['classes'], dtype=tf.float32)
            
            yield {"images": img, "bounding_boxes": {"boxes": boxes, "classes": classes}}

    dataset = tf_data.Dataset.from_generator(
        generator,
        output_signature={
            "images": tf.TensorSpec(shape=(640, 640, 3), dtype=tf.float32),
            "bounding_boxes": {
                "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                "classes": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            }
        }
    )
    
    # Padding cho batch (vì số lượng box mỗi ảnh khác nhau)
    dataset = dataset.ragged_batch(BATCH_SIZE, drop_remainder=True)
    return dataset.prefetch(AUTOTUNE)

# --- KHỞI TẠO DATASET ---
train_ds = make_dataset(
    image_dir='/work/Ner_Llm_Gpt/mediapipe/chunomdataset/train/images', 
    annotation_file='/work/Ner_Llm_Gpt/mediapipe/chunomdataset/train/labels.json'
)
val_ds = make_dataset(
    image_dir='/work/Ner_Llm_Gpt/mediapipe/chunomdataset/valid/images', 
    annotation_file='/work/Ner_Llm_Gpt/mediapipe/chunomdataset/valid/labels.json'
)


# 1. Khởi tạo Model (Dùng YOLOv8 Backbone có sẵn trong KerasCV)
# Model này nhẹ và mạnh tương đương MobileNet
model = keras_cv.models.YOLOV8Detector(
    num_classes=NUM_CLASSES, 
    bounding_box_format="xywh", # Khớp với format của COCO
    backbone=keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_m_backbone_coco"),
    fpn_depth=2 
)

# 2. Compile Model
# KerasCV tự động handle Box Loss và Classification Loss
optimizer = keras.optimizers.Adam(learning_rate=1e-3, global_clipnorm=10.0)
model.compile(
    optimizer=optimizer, 
    classification_loss="binary_crossentropy", 
    box_loss="ciou" # Loss hiện đại cho Object Detection
)

# 3. Cấu hình Callbacks (Cứu cánh khi Crash)
callbacks_list = [
    # A. ModelCheckpoint: Lưu file model (.keras) sau mỗi epoch để test inference
    keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/model_epoch_{epoch:02d}.keras',
        save_freq='epoch',          # Lưu sau mỗi epoch
        save_best_only=False,       # Lưu TẤT CẢ epoch (để bạn chọn lại cái tốt nhất)
        # save_weights_only=True    # Bật cái này nếu chỉ muốn lưu weights cho nhẹ
    ),
    
    # B. BackupAndRestore: Tự động lưu trạng thái tạm thời.
    # Nếu đang train bị mất điện/crash, chạy lại code này nó sẽ tự train tiếp từ epoch dở dang.
    keras.callbacks.BackupAndRestore(
        backup_dir="tmp/backup_training"
    ),
    
    # C. TensorBoard: Để theo dõi biểu đồ Loss
    keras.callbacks.TensorBoard(log_dir='logs')
]

# 4. Bắt đầu Train
# Không cần vòng lặp for thủ công, Keras lo hết
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,             # Set tổng số epoch mong muốn
    callbacks=callbacks_list
)

def inference():
    # Load model từ file đã lưu
    saved_model = keras.models.load_model('checkpoints/model_epoch_09.keras')

    # Hàm vẽ box (KerasCV hỗ trợ vẽ sẵn)
    def visualize_detection(image_path, model):
        # Load ảnh
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (640, 640))
        img_batch = tf.expand_dims(img, axis=0) # Thêm batch dimension -> (1, 640, 640, 3)

        # Dự đoán
        prediction = model.predict(img_batch)
        
        # KerasCV trả về dictionary: {'boxes': ..., 'classes': ..., 'confidence': ...}
        keras_cv.visualization.plot_bounding_box_gallery(
            img_batch,
            value_range=(0, 255),
            bounding_box_format="xywh",
            y_true=None,
            y_pred=prediction,
            scale=4,
            rows=1,
            cols=1
        )

    # Test thử
    visualize_detection("test_image.jpg", saved_model)


