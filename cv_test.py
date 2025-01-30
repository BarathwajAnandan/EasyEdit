import cv2


# Read the image
image = cv2.imread('../test.png')

# Define the parameters for the rectangle function
pt1 = (0, 50)  # Top-left corner
pt2 = (200, 200)  # Bottom-right corner
color = (0, 255, 0)  # Green color in BGR
thickness = 6  # Thickness of the rectangle border

# Call the rectangle function on the image
output_image = cv2.rectangle(image, pt1, pt2, color, thickness)

# Save the output image
cv2.imwrite('../output.png', output_image)
