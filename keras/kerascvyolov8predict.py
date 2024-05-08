import cv2
from keras.models import load_model
import tensorflow as tf

import keras_cv
import io

# Load the YOLOv8 model from the .h5 file
model = load_model('model.keras')

# if you want to use fastapi then upload image
# @webApp.post("/apis/vectorimage")
# async def vectorimage(image: UploadFile = File(...)):

#     imgbuff1 = await image.read()
#     bytes_object = io.BytesIO(imgbuff1).getvalue()
#     #Decode the JPEG bytes using tf.image.decode_jpeg
#     image_tensor = tf.image.decode_jpeg(bytes_object, channels=3)  # Specify the number of channels if known
#     tensor_data = tf.io.read_file(bytes_object)


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


imgpathfile = "/work/llm/Ner_Llm_Gpt/mediapipe/chunomdataset/valid/images/nlvnpf-0137-01-045.jpg"

image = load_image(imgpathfile)

height, width, channels = image.shape


image = tf.cast(image, tf.float32)

resizing = keras_cv.layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(1, 1),
    bounding_box_format="xyxy",
)


# # Apply resizing to the image
# Add batch dimension using tf.expand_dims
image = resizing(tf.expand_dims(image, 0))

# # # Remove the batch dimension if needed
# images = tf.squeeze(images, axis=0)

# image =tf.expand_dims(image, 0)

y_pred = model.predict(image)

print("boxes len", len(y_pred["boxes"].tolist()))
print("confidence len", len(y_pred["confidence"].tolist()))
print("classes len", len(y_pred["classes"].tolist()))


cv2img = cv2.imread(imgpathfile)

ih, iw, ic = cv2img.shape

for idx, bbss in enumerate(y_pred["boxes"].tolist()):
    for subidx, bb in enumerate(bbss):
        x = int(bb[0]*iw/640)
        y = int(bb[1]*ih/640)
        x1 = int(bb[2]*iw/640)
        y1 = int(bb[3]*ih/640)

        print("classes->", idx, y_pred["classes"].tolist()[idx][subidx])
        print("confidence->", idx, y_pred["confidence"].tolist()[idx][subidx])

        drawed = cv2.rectangle(
            cv2img, (x, y), (x1, y1), (0, 0, 255), 1)

        cv2.imshow("Image with Rectangle", drawed)
        cv2.waitKey(0)
