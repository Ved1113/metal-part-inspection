import cv2

img = cv2.imread("images/sample.jpg")

cv2.circle(img, (200, 300), 50, (0, 255, 0), 1)

cv2.imwrite("output/circle_sample.jpg", img)

cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()