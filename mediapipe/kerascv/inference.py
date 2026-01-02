
import os
import cv2
import numpy as np
import tensorflow as tf

# Define paths
EXPORTED_MODEL_DIR = "/work/Ner_Llm_Gpt/mediapipe/exported_model"
MODEL_NAME = "yolov8_chunom"
TFLITE_MODEL_PATH = os.path.join(EXPORTED_MODEL_DIR, f"{MODEL_NAME}.tflite")
TEST_IMAGE_PATH = "/work/Ner_Llm_Gpt/mediapipe/nlvnpf-0137-01-045.jpg"
RESULT_IMAGE_PATH = os.path.join(EXPORTED_MODEL_DIR, f"{MODEL_NAME}.test.result.jpg")

# --- Configuration ---
CONFIDENCE_THRESHOLD = 0.5 # Filter out detections with lower confidence
INPUT_SIZE = (640, 640) # The input size the TFLite model expects (YOLOv8 default)

def preprocess_image(image_path, input_size):
    """Loads and preprocesses an image for TFLite inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    original_shape = img.shape[:2] # (height, width)
    
    # Resize the image to the model's expected input size
    img_resized = cv2.resize(img, input_size)
    
    # Convert BGR to RGB, normalize, and add batch dimension
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32)
    # YOLO models often don't require normalization to [0,1] if they have a rescale layer,
    # but it's good practice. Let's assume the converted model handles it.
    
    return input_data, img, original_shape

def draw_boxes(image, boxes, scores, classes, original_shape, input_size):
    """Draws bounding boxes on the original image."""
    height, width = original_shape
    scale_x = width / input_size[0]
    scale_y = height / input_size[1]

    # The output of TFLite object detection models is often a set of arrays
    # representing boxes, classes, scores, and number of detections.
    # The exact format can vary. We'll assume a standard format here.
    
    # boxes: [1, N, 4] with (y_min, x_min, y_max, x_max) in range [0,1]
    # scores: [1, N]
    # classes: [1, N]
    
    for i in range(boxes.shape[1]):
        if scores[0, i] > CONFIDENCE_THRESHOLD:
            # Get box coordinates and denormalize
            y_min, x_min, y_max, x_max = boxes[0, i]
            
            # The TFLite output from KerasCV YoloV8 is often in xyxy format
            # and scaled to the input image size (e.g., 640x640)
            start_x = int(x_min * scale_x)
            start_y = int(y_min * scale_y)
            end_x = int(x_max * scale_x)
            end_y = int(y_max * scale_y)
            
            # Draw rectangle
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Prepare label text
            class_id = int(classes[0, i])
            score = scores[0, i]
            label = f"Class {class_id}: {score:.2f}"
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (start_x, start_y - 20), (start_x + w, start_y), (0, 255, 0), -1)
            # Draw label text
            cv2.putText(image, label, (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
    return image


def main():
    """Loads a TFLite model and performs inference on a test image."""
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"Error: TFLite model not found at {TFLITE_MODEL_PATH}")
        print("Please run 'convert.py' first.")
        return

    # Load the TFLite model and allocate tensors
    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image
    print(f"Loading and preprocessing image: {TEST_IMAGE_PATH}")
    input_data, original_image, original_shape = preprocess_image(TEST_IMAGE_PATH, INPUT_SIZE)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    print("Running inference...")
    interpreter.invoke()

    # The output format depends on the model conversion.
    # For KerasCV YOLOv8, the output might be a single tensor with shape [1, N, 5+num_classes]
    # containing [x, y, w, h, confidence, class1_prob, class2_prob, ...]
    # Or it could be multiple tensors. Let's check the output_details.
    
    # Assuming the output is from a `TFLiteDetectionPostProcess` op or similar,
    # which gives separated tensors for boxes, classes, scores.
    # We find the tensors by their names/indices.
    # This is a common pattern but might need adjustment.
    
    # Let's assume a simplified output for now: a dictionary of outputs.
    # The user can inspect `output_details` to map the correct indices.
    # A common output signature for TF OD models is:
    # output_details[0]: boxes
    # output_details[1]: classes
    # output_details[2]: scores
    # output_details[3]: num_detections
    
    # A more robust way is to inspect the output_details and find the correct tensors.
    # For a KerasCV YOLOv8 converted model, the output is typically a dictionary.
    # Let's assume the output is a list of tensors.
    # It's safer to get output by index.
    
    # Let's assume the converted model returns a dictionary like {'detection_boxes': ..., 'detection_scores': ...}
    # and we need to map names to indices.
    
    # The default converted KerasCV YoloV8 model has one output tensor
    # with boxes and scores combined.
    # The shape is [batch, num_boxes, 4 + num_classes]
    # Let's handle that logic.
    output = interpreter.get_tensor(output_details[0]['index'])

    # The output tensor combines boxes and classification results.
    # Boxes are in xywh format.
    boxes = output[0, :, :4]
    # The rest are confidence and class scores
    scores = output[0, :, 4:]
    
    # Find the class with the highest score for each detection
    class_ids = np.argmax(scores, axis=1)
    confidence_scores = np.max(scores, axis=1)

    # Filter based on confidence
    selected_indices = tf.image.non_max_suppression(
        boxes,
        confidence_scores,
        max_output_size=100,
        iou_threshold=0.5,
        score_threshold=CONFIDENCE_THRESHOLD
    )
    
    selected_boxes = tf.gather(boxes, selected_indices).numpy()
    selected_scores = tf.gather(confidence_scores, selected_indices).numpy()
    selected_classes = tf.gather(class_ids, selected_indices).numpy()
    
    # Convert xywh to xyxy for drawing
    x, y, w, h = selected_boxes[:, 0], selected_boxes[:, 1], selected_boxes[:, 2], selected_boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    
    # The coordinates are normalized to the input size (e.g., 640x640)
    final_boxes = np.stack([y1, x1, y2, x2], axis=1).reshape(1, -1, 4)
    final_scores = selected_scores.reshape(1, -1)
    final_classes = selected_classes.reshape(1, -1)
    
    print(f"Found {len(selected_boxes)} objects.")

    # Draw bounding boxes on the original image
    result_image = draw_boxes(original_image.copy(), final_boxes, final_scores, final_classes, original_shape, INPUT_SIZE)

    # Save the result
    cv2.imwrite(RESULT_IMAGE_PATH, result_image)
    print(f"Inference complete. Result saved to {RESULT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
