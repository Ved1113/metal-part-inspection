import cv2

# Read input image
img = cv2.imread("images/noisee.jpg")

if img is None:
    print("Error: Image not found.")
    exit()

# Gaussian Blur
gaussian = cv2.GaussianBlur(img, (21, 21), 0)

# Median Blur
median = cv2.medianBlur(img, 9)

# Bilateral Filter
bilateral = cv2.bilateralFilter(img, 9, 100, 100)

# Save outputs
cv2.imwrite("output/gaussian_blur.jpg", gaussian)
cv2.imwrite("output/median_blur.jpg", median)
cv2.imwrite("output/bilateral_blur.jpg", bilateral)

# Display results
cv2.imshow("Original", img)
cv2.imshow("Gaussian Blur", gaussian)
cv2.imshow("Median Blur", median)
cv2.imshow("Bilateral Filter", bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Filtering outputs saved successfully.")