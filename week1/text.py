import cv2

# Read image
img = cv2.imread("images/sample.jpg")

# Add text to image
cv2.putText(
    img,
    "helloo",
    (100, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

# Save output
cv2.imwrite("output/text_sample.jpg", img)

# Display image
cv2.imshow("Image with Text", img)

cv2.waitKey(0)
cv2.destroyAllWindows()