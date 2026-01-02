
import os
import tensorflow as tf

# Define paths
EXPORTED_MODEL_DIR = "/work/Ner_Llm_Gpt/mediapipe/exported_model"
MODEL_NAME = "yolov8_chunom"
# Path to the Keras model saved after training
SAVED_MODEL_PATH = os.path.join(EXPORTED_MODEL_DIR, f"{MODEL_NAME}_best.keras") 
# Path for the output TFLite model
TFLITE_MODEL_PATH = os.path.join(EXPORTED_MODEL_DIR, f"{MODEL_NAME}.tflite")

def main():
    """Loads a saved Keras model and converts it to TFLite format."""
    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"Error: Model not found at {SAVED_MODEL_PATH}")
        print("Please make sure you have trained the model using 'train.py' first.")
        return

    print(f"Loading Keras model from {SAVED_MODEL_PATH}...")
    # When loading a KerasCV model, it's often necessary to provide a custom_objects scope
    # However, for conversion, we can try loading without it first.
    # If that fails, you might need to specify the custom objects.
    try:
        model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    except Exception as e:
        print(f"Error loading model directly: {e}")
        print("This might be due to custom layers in KerasCV. A common workaround is to save in TF's saved_model format first.")
        # As a workaround, save to 'tf' format and reload
        temp_tf_format_path = os.path.join(EXPORTED_MODEL_DIR, "temp_tf_format")
        model_for_saving = tf.keras.models.load_model(SAVED_MODEL_PATH, compile=False)
        model_for_saving.save(temp_tf_format_path, save_format="tf")
        model = tf.keras.models.load_model(temp_tf_format_path)


    print("Converting model to TFLite...")

    # Initialize the TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations (e.g., quantize to float16 for smaller size and faster GPU inference)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert the model
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Successfully converted model to TFLite format.")
    print(f"TFLite model saved at: {TFLITE_MODEL_PATH}")

if __name__ == "__main__":
    main()
