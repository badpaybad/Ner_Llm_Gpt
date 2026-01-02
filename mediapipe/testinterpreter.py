

# STEP 1: Import the necessary modules.
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

import tensorflow as tf

import cv2

# Load the network
# net = cv2.dnn.readNetFromMNN("/work/llm/Ner_Llm_Gpt/mediapipe/face_detect_live/model_1.bin", "/work/llm/Ner_Llm_Gpt/mediapipe/face_detect_live/model_1.param")
# print(net)
# Prepare input data
# blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(224, 224))

# Set input and run forward pass
# net.setInput(blob)
# output = net.forward()


def check_tflite_in_out_info(modelpathfile):
    modelpathfile="/work/llm/Ner_Llm_Gpt/mediapipe/face_detect_live/model_1.bin"
    modelcontent="/work/llm/Ner_Llm_Gpt/mediapipe/face_detect_live/model_1.param"
    print(modelpathfile)
    # Load the TFLite model
    interpreter = tf.lite.Interpreter( model_content=modelpathfile)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the tensor to point to the input data
    print("-------------------set input random value")
    # can convert image into input numpy array
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    # interpreter.set_tensor(input_details[0]['index'], input_data)
    
# check_tflite_in_out_info("/work/llm/Ner_Llm_Gpt/mediapipe/face_detect_live/model_1.bin")