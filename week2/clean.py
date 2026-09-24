import cv2
import numpy as np

img = cv2.imread("images/threshoo.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Upscale first - gives more pixel detail for text to survive processing
upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Denoise
denoised = cv2.fastNlMeansDenoising(upscaled, None, h=20, templateWindowSize=7, searchWindowSize=21)

# Adaptive threshold (blockSize scaled up since image is bigger now)
thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 41, 5)

# Light morphological cleanup
kernel = np.ones((2,2), np.uint8)
final = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Final Cleaned Document', final)
cv2.imwrite('final_cleanedd.jpg', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('final_cleaned.jpg', final)