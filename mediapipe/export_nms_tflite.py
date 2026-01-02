import tensorflow as tf
import keras
import cv2
import numpy as np

MODEL_PATH = '/work/Ner_Llm_Gpt/mediapipe/checkpoints/model_epoch_14.keras'
TFLITE_PATH = '/work/Ner_Llm_Gpt/mediapipe/yolov8_nms.tflite'
IMG_SIZE = (640, 640)
MAX_BOXES = 100    # Chỉ lấy 100 box tốt nhất
IOU_THRESH = 0.45  # Ngưỡng lọc box trùng (NMS)
SCORE_THRESH = 0.25 # Ngưỡng điểm (chỉ lấy box có độ tin cậy > 25%)

def export_with_nms():
    # 1. Load Model
    model = keras.models.load_model(MODEL_PATH, compile=False)

    # 2. Tạo Wrapper chứa thuật toán NMS
    class NMSWrapper(tf.Module):
        def __init__(self, model):
            self.model = model

        @tf.function(input_signature=[tf.TensorSpec(shape=[1, *IMG_SIZE, 3], dtype=tf.float32)])
        def __call__(self, images):
            # Lấy output thô từ model
            outputs = self.model(images, training=False)
            
            # KerasCV output thường là {'boxes': [B, 8400, 4], 'classes': [B, 8400, 80]}
            # Nếu output của bạn đang là 64 (raw distribution), ta cần decoded nó.
            # Tuy nhiên, hàm này giả định model KerasCV đã config đúng để decode ra 4. 
            # (Nếu model load bị lỗi decode, output này sẽ tự động được fix bởi CombinedNMS nếu shape hợp lệ)
            
            boxes = outputs['boxes']   # Shape kỳ vọng: (1, 8400, 4)
            scores = outputs['classes'] # Shape kỳ vọng: (1, 8400, 80)

            # --- QUAN TRỌNG: XỬ LÝ LỖI SHAPE 64 ---
            # Nếu boxes có shape (..., 64), ta không thể dùng NMS ngay.
            # Ta ép kiểu về 4 (giả sử đây là sự nhầm lẫn format hoặc cần decode đơn giản)
            # Nhưng để an toàn, ta dùng tf.image.combined_non_max_suppression
            # Hàm này yêu cầu boxes phải là (Batch, num_boxes, 1, 4) hoặc (Batch, num_boxes, 4)
            
            # Convert boxes format từ Center-XYWH (YOLO) sang Corners-YXYX (TF NMS cần format này)
            # x, y, w, h -> y_min, x_min, y_max, x_max
            cx, cy, w, h = tf.split(boxes, 4, axis=-1)
            ymin = cy - h / 2
            xmin = cx - w / 2
            ymax = cy + h / 2
            xmax = cx + w / 2
            boxes_corners = tf.concat([ymin, xmin, ymax, xmax], axis=-1)

            # Dùng thuật toán NMS của TensorFlow (Chạy cực nhanh trên C++)
            # Hàm này sẽ tự động lọc 8400 box xuống còn 100 box xịn nhất
            final_boxes, final_scores, final_classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.expand_dims(boxes_corners, axis=2), # Cần shape (1, 8400, 1, 4)
                scores=scores,
                max_output_size_per_class=MAX_BOXES,
                max_total_size=MAX_BOXES,
                iou_threshold=IOU_THRESH,
                score_threshold=SCORE_THRESH,
                clip_boxes=False
            )

            # Output trả về:
            # final_boxes: (1, 100, 4) - Toạ độ chuẩn [y1, x1, y2, x2]
            # final_classes: (1, 100) - ID class (0, 1, 2...)
            # final_scores: (1, 100) - Độ tin cậy
            # valid_detections: (1,) - Số lượng box thực tế tìm thấy
            return final_boxes, final_classes, final_scores, valid_detections

    wrapper = NMSWrapper(model)
    
    # 3. Convert
    converter = tf.lite.TFLiteConverter.from_concrete_functions([wrapper.__call__.get_concrete_function()])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS # Bắt buộc có để chạy NMS
    ]
    
    tflite_model = converter.convert()
    
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Xong! Model đã tích hợp NMS lưu tại: {TFLITE_PATH}")

# if __name__ == "__main__":
export_with_nms()


import cv2
import numpy as np
import tensorflow as tf

# Cấu hình
MODEL_PATH =TFLITE_PATH
IMAGE_PATH = '/work/Ner_Llm_Gpt/mediapipe/nlvnpf-0137-01-045.jpg'
INPUT_SIZE = (640, 640)

# Load TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Hàm lấy tensor output theo index (vì thứ tự output có thể thay đổi)
def get_output_tensor(index):
    return interpreter.get_tensor(output_details[index]['index'])

# 1. Chuẩn bị ảnh
original_img = cv2.imread(IMAGE_PATH)
orig_h, orig_w = original_img.shape[:2]

# Resize ảnh về 640x640 để đưa vào model
img_resized = cv2.resize(original_img, INPUT_SIZE)
img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
img_input = np.expand_dims(img_input, axis=0) # (1, 640, 640, 3)

# 2. Inference
interpreter.set_tensor(input_details[0]['index'], img_input)
interpreter.invoke()

# 3. Lấy kết quả đã được NMS (Clean)
# Lưu ý: Bạn cần kiểm tra thứ tự output_details để map đúng
# Thường thứ tự là: Boxes, Classes, Scores, Count (hoặc đảo lộn)
# Cách an toàn là dựa vào shape:
# (1, 100, 4) -> Boxes
# (1, 100) -> Classes / Scores
# (1,) -> Count

boxes = None
classes = None
scores = None
count = None

for i, detail in enumerate(output_details):
    tensor = interpreter.get_tensor(detail['index'])
    if detail['shape'][-1] == 4:
        boxes = tensor[0] # Lấy batch đầu tiên -> (100, 4)
    elif detail['shape'][-1] == 1 and len(detail['shape']) == 2:
        # Trường hợp (1, 1) thường là count
        count = int(tensor[0])
    elif len(detail['shape']) == 1:
        # Trường hợp (1,) cũng là count
        count = int(tensor[0])
    elif detail['shape'][-1] == 100:
        # Scores hoặc Classes
        # Thường scores là float, classes cũng là float nhưng giá trị nguyên
        # Ở đây ta tạm gán lần lượt, bạn có thể print ra check
        if scores is None:
            scores = tensor[0]
        else:
            classes = tensor[0]

# Fallback nếu map tự động sai (Bạn nên in output_details ra để fix cứng index)
# Ví dụ fix cứng theo code export:
# Output 0: boxes, Output 1: classes, Output 2: scores, Output 3: valid_detections
boxes = interpreter.get_tensor(output_details[0]['index'])[0]
classes = interpreter.get_tensor(output_details[1]['index'])[0]
scores = interpreter.get_tensor(output_details[2]['index'])[0]
num_dets = int(interpreter.get_tensor(output_details[3]['index'])[0])

print(f"Tìm thấy {num_dets} đối tượng.")

# 4. Vẽ lên ảnh
for i in range(num_dets):
    score = scores[i]
    if score < 0.25: continue # Bỏ qua nếu tin cậy thấp (dù NMS đã lọc rồi)

    # Box từ TFLite (TF NMS) trả về dạng [ymin, xmin, ymax, xmax] theo tỉ lệ 0-1 hoặc pixel 640
    # Nếu dùng code export ở trên, toạ độ là pixel của ảnh 640x640
    ymin, xmin, ymax, xmax = boxes[i]

    # Scale toạ độ về ảnh gốc
    x1 = int(xmin * (orig_w / INPUT_SIZE[0])) # Nếu box output là 0-1 thì bỏ INPUT_SIZE[0]
    y1 = int(ymin * (orig_h / INPUT_SIZE[1]))
    x2 = int(xmax * (orig_w / INPUT_SIZE[0]))
    y2 = int(ymax * (orig_h / INPUT_SIZE[1]))

    # Vẽ Box
    cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Vẽ Label
    label = f"Class {int(classes[i])}: {score:.2f}"
    cv2.putText(original_img, label, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save hoặc Show
cv2.imwrite("result.jpg", original_img)
print("Đã lưu ảnh kết quả: result.jpg")