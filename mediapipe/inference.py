
# inference https://github.com/google-edge-ai/mediapipe-samples/blob/main/examples/object_detection/python/object_detector.ipynb


# STEP 1: Import the necessary modules.
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image


# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(
    model_asset_path='/work/llm/Ner_Llm_Gpt/mediapipe/exported_model/model.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)
print("STEP 2: Create an ObjectDetector object.")
# STEP 3: Load the input image.
image = mp.Image.create_from_file(
    # "/home/dunp/Downloads/android_figurine/train/images/IMG_0509.jpg"
    # "/work/cloud/cloud.mldlai/test/new.jpg.txt.jpg"
    "/work/llm/Ner_Llm_Gpt/mediapipe/nlvnpf-0137-01-045.jpg"
)
print("# STEP 3: Load the input image.")
# STEP 4: Detect objects in the input image.
detection_result = detector.detect(image)
print("# STEP 4: Detect objects in the input image.")
# STEP 5: Process the detection result. In this case, visualize it.
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
# cv2.imshow("", rgb_annotated_image)
print("STEP 5: Process the detection result. In this case, visualize it.")

cv2.imwrite("detected.jpg", rgb_annotated_image)

print("STEP 6: Save detected image")

cv2.imshow("Image", rgb_annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
# import matplotlib
# import matplotlib.rcsetup as rcsetup
# print(matplotlib.matplotlib_fname())
# print(rcsetup.all_backends)
# guibackend = "Qt5Agg"
# matplotlib.use(guibackend)
# plt.switch_backend(guibackend)
# # # plt.style.use('ggplot')

# matplotlib.rcParams['backend'] = guibackend
# print(f"Interactive mode: {matplotlib.is_interactive()}")
# print(f"matplotlib backend: {matplotlib.rcParams['backend']}")
# imgplot = plt.imshow(rgb_annotated_image, cv2.WINDOW_GUI_EXPANDED)
# plt.show()

