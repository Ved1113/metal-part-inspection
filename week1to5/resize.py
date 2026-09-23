import cv2
img = cv2.imread("images/sample.jpg")
resized=cv2.resize(img, (500,500))
cv2.imshow("Image", resized)
cv2.imshow("Original",img)
print("Shape:", resized.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()