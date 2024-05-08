from keras.models import load_model
import tensorflow  as tf

import keras_cv,io

# Load the YOLOv8 model from the .h5 file
model = load_model('model.keras')

## if you want to use fastapi then upload image 
# @webApp.post("/apis/vectorimage")
# async def vectorimage(image: UploadFile = File(...)):

#     imgbuff1 = await image.read()
#     bytes_object = io.BytesIO(imgbuff1).getvalue()
#     tensor_data = tf.io.read_file(bytes_object)

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

imgpathfile="/work/llm/Ner_Llm_Gpt/mediapipe/chunomdataset/valid/images/nlvnpf-0137-01-045.jpg"

image = load_image(imgpathfile)

tf.cast(image, tf.float32)

resizing = keras_cv.layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(0.75, 1.3),
    bounding_box_format="xyxy",
)


# Apply resizing to the image
images = resizing(tf.expand_dims(image, 0))  # Add batch dimension using tf.expand_dims

# # # Remove the batch dimension if needed
# images = tf.squeeze(images, axis=0)

y_pred = model.predict(images)

print(y_pred["boxes"].tolist()[0])
print(y_pred["confidence"].tolist()[0])
print(y_pred["classes"].tolist()[0])

import cv2

cv2img= cv2.imread(imgpathfile)

for idx,bb in enumerate(y_pred["boxes"].tolist()[0]):
    x=int(bb[0])
    y=int(bb[1])
    x1=int(bb[2])
    y1=int(bb[3])
    
    print("classes->",y_pred["classes"].tolist()[0][idx])
    print("confidence->",y_pred["confidence"].tolist()[0][idx])
        
    drawed = cv2.rectangle(
                    cv2img, (x, y), (x1, y1), (0, 255, 0), 1)
              
    cv2.imshow("Image with Rectangle", drawed)
    cv2.waitKey(0)
    
    