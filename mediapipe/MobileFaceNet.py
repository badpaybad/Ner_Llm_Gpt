import tensorflow as tf
import numpy as np
import cv2

def get_embedding(face_crop, model_path):
    # 1. Load model TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Tiền xử lý ảnh khuôn mặt (đã crop từ detector của bạn)
    # Giả sử model yêu cầu 112x112
    face_img = cv2.resize(face_crop, (112, 112))
    face_img = face_img.astype(np.float32)
    face_img = (face_img - 127.5) / 128.0  # Chuẩn hóa về [-1, 1]
    face_img = np.expand_dims(face_img, axis=0)

    # 3. Chạy Inference
    interpreter.set_tensor(input_details[0]['index'], face_img)
    interpreter.invoke()

    # 4. Lấy vector embedding
    embedding = interpreter.get_tensor(output_details[0]['index'])
    return embedding.flatten()