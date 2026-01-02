import tensorflow as tf
import keras

def convert_to_tflite_quantized(keras_model_path, tflite_output_path):
    model = keras.models.load_model(keras_model_path, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Bật cờ tối ưu hóa
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Hỗ trợ cả Built-in và TF Ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    tflite_model = converter.convert()

    with open(tflite_output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Đã save model tối ưu hóa: {tflite_output_path}")

# Chạy
# convert_to_tflite_quantized('/work/Ner_Llm_Gpt/mediapipe/checkpoints/model_epoch_09.keras', '/work/Ner_Llm_Gpt/mediapipe/yolov8_quantized.tflite')


import numpy as np
import tensorflow as tf
import cv2

# Đường dẫn file tflite
TFLITE_MODEL_PATH = '/work/Ner_Llm_Gpt/mediapipe/yolov8_quantized.tflite'
IMAGE_PATH = 'nlvnpf-0137-01-045.jpg'

# 1. Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Lấy thông tin input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Shape:", input_details[0]['shape']) 
# Output thường trả về nhiều tensor (Boxes, Classes, Confidence...)

# 2. Chuẩn bị ảnh
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) # Model float input
img = np.expand_dims(img, axis=0) # (1, 640, 640, 3)

# 3. Đưa ảnh vào model
interpreter.set_tensor(input_details[0]['index'], img)

# 4. Chạy dự đoán
interpreter.invoke()

# 5. Lấy kết quả
# Lưu ý: Thứ tự output của KerasCV export sang TFLite có thể thay đổi.
# Bạn cần in output_details để xem index nào là boxes, index nào là classes.
for i, detail in enumerate(output_details):
    data = interpreter.get_tensor(detail['index'])
    print(f"Output {i} ({detail['name']}): shape={data.shape}")
    # Ví dụ xử lý: nếu shape là (1, N, 4) thì đó là boxes


import tensorflow as tf
import keras_cv
import keras

# Đảm bảo cài đặt các tham số giống lúc train
IMAGE_SIZE = (640, 640)

def convert_to_tflite_float32(keras_model_path, tflite_output_path):
    print(f"Đang load model từ: {keras_model_path}")
    # Load model gốc
    # Lưu ý: compile=False giúp load nhanh hơn nếu không cần train tiếp
    model = keras.models.load_model(keras_model_path, compile=False)

    print("Đang khởi tạo Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # --- QUAN TRỌNG: Xử lý các Ops phức tạp của YOLO ---
    # Vì YOLOv8 dùng một số hàm (như NMS) có thể chưa được hỗ trợ native trong TFLite chuẩn,
    # ta nên cho phép sử dụng thêm các Ops của TensorFlow (Flex Delegate) nếu cần.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Ops chuẩn của TFLite
        tf.lite.OpsSet.SELECT_TF_OPS    # Ops của TF (giúp tránh lỗi nếu TFLite thiếu ops)
    ]
    
    # Ép kiểu input shape cố định (TFLite thích input cố định hơn dynamic)
    # Model KerasCV đôi khi để None, nên ta define rõ
    # (Nếu model đã fix shape rồi thì bước này converter tự hiểu)
    
    print("Đang convert...")
    tflite_model = converter.convert()

    # Lưu file
    with open(tflite_output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Hoàn tất! File saved tại: {tflite_output_path}")

# # Chạy thử
# convert_to_tflite_float32('checkpoints/model_epoch_09.keras', 'yolov8_detector.tflite')