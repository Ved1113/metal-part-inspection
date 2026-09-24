import cv2
import numpy as np

img = cv2.imread("images/threshoo.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

#blurred = cv2.GaussianBlur(img, (21, 21), 0)
#median_blurred = cv2.medianBlur(gray, 9)
#bilateral = cv2.bilateralFilter(img, 9, 100, 100)
# Strong denoising - better than median blur for heavy noise
denoised = cv2.fastNlMeansDenoising(gray, None, h=25, templateWindowSize=7, searchWindowSize=21)


thresh_5 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 25, 3)
kernel = np.ones((2,2), np.uint8)
final = cv2.morphologyEx(thresh_5, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Final Cleaned Document', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output/final_cleaned.jpg", final)