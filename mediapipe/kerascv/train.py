
import os
import tensorflow as tf
import keras_cv
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint

# Define paths
TRAIN_IMAGES_DIR = "/work/Ner_Llm_Gpt/mediapipe/chunomdataset/train/images"
TRAIN_ANNOTATIONS_PATH = "/work/Ner_Llm_Gpt/mediapipe/chunomdataset/train/labels.json"
VALID_IMAGES_DIR = "/work/Ner_Llm_Gpt/mediapipe/chunomdataset/valid/images"
VALID_ANNOTATIONS_PATH = "/work/Ner_Llm_Gpt/mediapipe/chunomdataset/valid/labels.json"

EXPORTED_MODEL_DIR = "/work/Ner_Llm_Gpt/mediapipe/exported_model"
LOGS_DIR = "/work/Ner_Llm_Gpt/mediapipe/logs"
MODEL_NAME = "yolov8_chunom"

# Create directories if they don't exist
os.makedirs(EXPORTED_MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Configuration ---
BATCH_SIZE = 4
EPOCHS = 20
# Use a smaller preset for faster training/demonstration if needed
# For example: "yolo_v8_xs_pascalvoc"
PRESET = "yolo_v8_m_pascalvoc" 

def load_data(image_dir, annotation_path, batch_size):
    """Loads a COCO dataset and prepares it for training."""
    return keras_cv.loaders.load_coco(
        image_dir=image_dir,
        annotation_path=annotation_path,
        batch_size=batch_size,
        bounding_box_format="xywh"
    )

def main():
    """Main function to set up and run the training process."""
    # Load datasets
    print("Loading training data...")
    train_ds = load_data(TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_PATH, BATCH_SIZE)
    print("Loading validation data...")
    valid_ds = load_data(VALID_IMAGES_DIR, VALID_ANNOTATIONS_PATH, BATCH_SIZE)

    # Get the number of classes from the dataset
    # This assumes the class IDs in the COCO file are contiguous from 0
    # You might need to adjust this based on your actual `labels.json`
    # For this example, let's assume there is 1 class.
    # Replace `num_classes=1` with the actual number of classes in your dataset.
    NUM_CLASSES = 1 

    # Create the model
    print(f"Creating model with preset: {PRESET}")
    model = keras_cv.models.YOLOV8Detector.from_preset(
        PRESET,
        num_classes=NUM_CLASSES,
        bounding_box_format="xywh"
    )

    # Compile the model
    print("Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        box_loss="ciou",
        classification_loss="binary_crossentropy", # Use 'categorical_crossentropy' for multi-class
    )

    # Define callbacks
    tensorboard_callback = TensorBoard(log_dir=LOGS_DIR)
    csv_logger_callback = CSVLogger(os.path.join(LOGS_DIR, f"{MODEL_NAME}_training_log.csv"))
    # Saves the model with the best validation box loss
    model_checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(EXPORTED_MODEL_DIR, f"{MODEL_NAME}_best.keras"),
        monitor="val_box_loss",
        save_best_only=True,
        mode="min"
    )
    
    print("Starting training...")
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=[
            tensorboard_callback,
            csv_logger_callback,
            model_checkpoint_callback
        ]
    )
    
    # Save the final model
    final_model_path = os.path.join(EXPORTED_MODEL_DIR, f"{MODEL_NAME}_final.keras")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
