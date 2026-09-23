import cv2

# Read the image
img = cv2.imread("images/sample.jpg")

# Crop the image
cropped = img[100:400, 200:600]

# Display original and cropped images
cv2.imshow("Original Image", img)
cv2.imshow("Cropped Image", cropped)


# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()