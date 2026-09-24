import cv2
import numpy as np

# Read noisy document
img = cv2.imread("images/threshoo.jpg")

if img is None:
    print("Error: Image not found.")
    exit()

# 1. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Remove noise using median filter
denoised = cv2.medianBlur(gray, 5)

# 3. Adaptive threshold
thresh = cv2.adaptiveThreshold(
    denoised,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    25,
    5
)

# 4. Morphological opening
kernel = np.ones((2, 2), np.uint8)

opened = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel
)

# 5. Morphological closing
final = cv2.morphologyEx(
    opened,
    cv2.MORPH_CLOSE,
    kernel
)

# Save each stage
cv2.imwrite("output/document_gray.jpg", gray)
cv2.imwrite("output/document_denoised.jpg", denoised)
cv2.imwrite("output/document_threshold.jpg", thresh)
cv2.imwrite("output/document_opening.jpg", opened)
cv2.imwrite("output/document_final.jpg", final)

# Display final result
cv2.imshow("Original", img)
cv2.imshow("Final Cleaned Document", final)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Final document cleaning completed.")