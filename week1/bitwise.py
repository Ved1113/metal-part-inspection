import cv2
import numpy as np

img = cv2.imread("images/sample.jpg")

mask = np.zeros(img.shape[:2], dtype=np.uint8)

cv2.rectangle(mask, (200, 150), (600, 400), 255, -1)

result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Original", img)
cv2.imshow("Mask", mask)

cv2.imwrite("output/bitwise_result.jpg", result)

cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()