import cv2

# Read image
img = cv2.imread("images/sample.jpg")

# Select ROI
roi = img[200:500, 300:700]

# Show original and ROI
cv2.imshow("Original Image", img)
cv2.imshow("ROI", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()