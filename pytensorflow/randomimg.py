import numpy as np
import matplotlib.pyplot as plt

# Generating a random grayscale image of shape (28, 28, 1)
random_image = np.random.rand(2, 3, 1)  # Creating a random image array
random_image = random_image * 255  # Scaling to pixel values (0-255)

print(random_image)
print(random_image.squeeze())

# Display the image using matplotlib
plt.imshow(random_image,)  # Squeezing to remove the single channel dimension
plt.axis('off')  # Hide axis labels
plt.title('Example Image (28x28)')
plt.show()
