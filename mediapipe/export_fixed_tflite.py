import os
import tensorflow as tf
import keras
import keras_cv
import numpy as np

# Cấu hình
MODEL_PATH = '/work/Ner_Llm_Gpt/mediapipe/checkpoints/model_epoch_14.keras' # Đường dẫn file model của bạn
TFLITE_PATH = '/work/Ner_Llm_Gpt/mediapipe/yolov8_fixed.tflite'
IMG_SIZE = (640, 640)

def export_model():
    # 1. Load model gốc
    # compile=False để tránh load optimizer state gây nặng và lỗi không cần thiết
    print("Loading Keras model...")
    model = keras.models.load_model(MODEL_PATH, compile=False)

    # 2. Tạo Wrapper Class để CỐ ĐỊNH input shape và output format
    # TFLite rất ghét dictionary output của KerasCV, nên ta chuyển thành tuple
    class YOLOv8Wrapper(tf.Module):
        def __init__(self, model):
            self.model = model

        @tf.function(input_signature=[tf.TensorSpec(shape=[1, *IMG_SIZE, 3], dtype=tf.float32)])
        def __call__(self, images):
            # Gọi model
            outputs = self.model(images, training=False)
            
            # Tách dictionary {'boxes': ..., 'classes': ...} thành danh sách tensor rời
            # Điều này giúp TFLite hiểu rõ output node nào là gì
            return outputs['boxes'], outputs['classes']

    print("Wrapping model...")
    wrapper = YOLOv8Wrapper(model)

    # 3. Convert từ Concrete Function (Quan trọng nhất)
    # Lấy concrete function để "đóng băng" graph với kích thước (1, 640, 640, 3)
    concrete_func = wrapper.__call__.get_concrete_function()

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    # Cấu hình Ops hỗ trợ (YOLOv8 cần TF Ops cho phần giải mã Box)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Ops chuẩn
        tf.lite.OpsSet.SELECT_TF_OPS    # Ops nâng cao của TF (cần thiết cho YOLO)
    ]
    
    # Tắt việc check experimental để tránh warning
    converter.experimental_new_converter = True

    # (Tùy chọn) Bật Quantization để giảm dung lượng file
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # 4. Lưu file
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Success! Model saved to {TFLITE_PATH}")

# if __name__ == "__main__":
export_model()

import tensorflow as tf
import cv2
import numpy as np

# Load model
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors() # Bước này sẽ không bị lỗi nữa

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print để xem index của output (thường output 0 là boxes, 1 là classes hoặc ngược lại)
print("Output Info:", output_details)

# Chuẩn bị ảnh test
img = cv2.imread("/work/Ner_Llm_Gpt/mediapipe/nlvnpf-0137-01-045.jpg")
img = cv2.resize(img, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
img = np.expand_dims(img, axis=0) # Shape (1, 640, 640, 3)

# Run Inference
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

# Lấy kết quả
# Lưu ý: Bạn cần nhìn vào `output_details` in ra ở trên để biết index nào là boxes
# Giả sử index 0 là output đầu tiên trong return của wrapper (boxes)
boxes = interpreter.get_tensor(output_details[0]['index']) 
classes = interpreter.get_tensor(output_details[1]['index'])

print("Boxes shape:", boxes.shape)     # Mong đợi: (1, N, 4)
print("Classes shape:", classes.shape) # Mong đợi: (1, N)