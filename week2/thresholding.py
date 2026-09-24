import cv2

# Read image
img = cv2.imread("images/sudoku.png")

if img is None:
    print("Error: Image not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Simple Threshold
_, simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 2. Adaptive Threshold
adaptive = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

# 3. Otsu Threshold
_, otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Save outputs
cv2.imwrite("output/simple_threshold.jpg", simple)
cv2.imwrite("output/adaptive_threshold.jpg", adaptive)
cv2.imwrite("output/otsu_threshold.jpg", otsu)

# Display
cv2.imshow("Original Grayscale", gray)
cv2.imshow("Simple Threshold", simple)
cv2.imshow("Adaptive Threshold", adaptive)
cv2.imshow("Otsu Threshold", otsu)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Thresholding outputs saved successfully.")