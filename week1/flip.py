import cv2

# Read image
img = cv2.imread("images/sample.jpg")

# Flip image
horizontal = cv2.flip(img, 1)
vertical = cv2.flip(img, 0)

# Save outputs
cv2.imwrite("output/flip_horizontal.jpg", horizontal)
cv2.imwrite("output/flip_vertical.jpg", vertical)

# Display
cv2.imshow("Original", img)
cv2.imshow("Horizontal Flip", horizontal)
cv2.imshow("Vertical Flip", vertical)

cv2.waitKey(0)
cv2.destroyAllWindows()