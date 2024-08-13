import cv2
import numpy as np

# Load the source and target images
source_image = cv2.imread('/work/llm/Ner_Llm_Gpt/deepfacefake/areafake.png')
target_image = cv2.imread('/work/llm/Ner_Llm_Gpt/deepfacefake/keeped.png')

# Define corresponding points in the source and target images
# Example points: these should be manually chosen or detected using feature matching
src_pts = np.array([[50, 50], [200, 50], [50, 200], [200, 200]], dtype='float32')  # Points in source image
dst_pts = np.array([[30, 60], [180, 40], [60, 230], [210, 220]], dtype='float32')  # Corresponding points in target image

# Compute the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the perspective transformation to the target image
warped_image = cv2.warpPerspective(target_image, M, (source_image.shape[1], source_image.shape[0]))

# Show the images
cv2.imshow('Source Image', source_image)
cv2.imshow('Target Image', target_image)
cv2.imshow('Warped Target Image', warped_image)

# Save the result
cv2.imwrite('warped_image.jpg', warped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
