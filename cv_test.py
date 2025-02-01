import cv2
import numpy as np

# Read the image
image = cv2.imread('../test.png')
# CONVERT TO GRAYSCALE
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Call the HoughCircles function on the image
circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.5, 10, param1=100, param2=0.7, minRadius=4, maxRadius=30)

# Ensure the output is in a format suitable for saving
if circles is not None:
    circles = np.uint16(np.around(circles))

# Create a copy of the original image to draw circles
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw the circles on the output image
if circles is not None:
    for i in circles[0, :]:
        # Draw the outer circle
        cv2.circle(output_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(output_image, (i[0], i[1]), 2, (0, 0, 255), 3)

# Save the output image
cv2.imwrite('../output.png', output_image)
