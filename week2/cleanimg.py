import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

img = cv2.imread("images/threshoo.jpg")
print("Image loaded:", img is not None)
print("Image shape:", img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

denoised = cv2.fastNlMeansDenoising(gray, None, h=25, templateWindowSize=7, searchWindowSize=21)
print("Denoised - min:", denoised.min(), "max:", denoised.max(), "mean:", denoised.mean())

thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 25, 3)
print("Thresh - unique values:", np.unique(thresh))
print("Thresh - white pixel %:", (thresh == 255).sum() / thresh.size * 100)

kernel = np.ones((2,2), np.uint8)
final = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('final_cleaned_TEST.jpg', final)
print("Saved final_cleaned_TEST.jpg")
print("DONE - if you see this, script ran completely without errors")