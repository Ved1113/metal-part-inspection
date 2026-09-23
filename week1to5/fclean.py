import cv2
import numpy as np

img = cv2.imread("images/noisee.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Upscale for more pixel detail
upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

# 2. Denoise
denoised = cv2.fastNlMeansDenoising(upscaled, None, h=15, templateWindowSize=7, searchWindowSize=21)

# 3. Sharpen slightly to crisp up letter edges
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

# 4. Adaptive threshold
thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 51, 6)

# 5. Remove small noise blobs using connected components (better than basic morphology here)
inverted = cv2.bitwise_not(thresh)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

min_size = 40
cleaned = np.zeros_like(inverted)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_size:
        cleaned[labels == i] = 255

final = cv2.bitwise_not(cleaned)

cv2.imshow('Final Cleaned Document', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('final_clean.jpg', final)