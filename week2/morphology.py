import cv2
import numpy as np

# Read image
img = cv2.imread("images/sudoku.png")

if img is None:
    print("Error: Image not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to binary image
_, binary = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)

# Kernel
kernel = np.ones((3, 3), np.uint8)

# 1. Erosion
erosion = cv2.erode(binary, kernel, iterations=1)

# 2. Dilation
dilation = cv2.dilate(binary, kernel, iterations=1)

# 3. Opening
opening = cv2.morphologyEx(
    binary, cv2.MORPH_OPEN, kernel
)

# 4. Closing
closing = cv2.morphologyEx(
    binary, cv2.MORPH_CLOSE, kernel
)

# Save outputs
cv2.imwrite("output/erosion.jpg", erosion)
cv2.imwrite("output/dilation.jpg", dilation)
cv2.imwrite("output/opening.jpg", opening)
cv2.imwrite("output/closing.jpg", closing)

# Display
cv2.imshow("Original Binary", binary)
cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Morphology outputs saved successfully.")