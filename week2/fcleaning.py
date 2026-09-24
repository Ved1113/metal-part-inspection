import cv2
import numpy as np

# ---- STEP 1: Load and convert to grayscale ----
img = cv2.imread("images/threshoo.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Upscale the image
upscaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

#  Denoise using Non-Local Means 
denoised = cv2.fastNlMeansDenoising(upscaled, None, h=30, 
                                      templateWindowSize=7, searchWindowSize=21)

#  Adaptive thresholding 
thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 61, 7)

#  Remove leftover noise using connected components 
inverted = cv2.bitwise_not(thresh)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

min_size = 130
cleaned = np.zeros_like(inverted)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_size:
        cleaned[labels == i] = 255

#  Light morphological closing 
kernel = np.ones((2,2), np.uint8)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

#  Invert back and save 
final = cv2.bitwise_not(cleaned)
cv2.imshow('Final Cleaned Document', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('cleanfinal.jpg', final)