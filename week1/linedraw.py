import cv2

img = cv2.imread("images/sample.jpg")

if img is None:
    raise FileNotFoundError("Could not read images/sample.jpg")

# Draw a diagonal line on the image.
cv2.line(img, (50, 50), (500, 300), (255, 0, 0), 4)

cv2.imwrite("output/line_sample.jpg", img)
print("Saved output/line_sample.jpg")

cv2.imshow("Line", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
