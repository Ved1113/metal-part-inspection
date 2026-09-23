import cv2

# Read image
img = cv2.imread("images/sample.jpg")

# Rotate image 90 degree clockwise
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Display original and rotated image
cv2.imshow("Original Image", img)
cv2.imshow("Rotated Image", rotated)

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()