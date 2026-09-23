import cv2
import numpy as np

# Load image
img = cv2.imread("images/sample.jpg")

if img is None:
    print("Image not found!")
    exit()

# Split into B, G, R channels
B, G, R = cv2.split(img)

# Display channels
cv2.imshow("Original", img)
cv2.imshow("Blue Channel", B)
cv2.imshow("Green Channel", G)
cv2.imshow("Red Channel", R)

# Merge channels back
merged = cv2.merge([B, G, R])

cv2.imshow("Merged", merged)

# Compare original and merged
if np.array_equal(img, merged):
    print("Merged image matches the original!")
else:
    print("Merged image does NOT match the original.")

cv2.waitKey(0)
cv2.destroyAllWindows()