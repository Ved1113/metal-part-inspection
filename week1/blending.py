import cv2

img1 = cv2.imread("images/sample.jpg")

if img1 is None:
    raise FileNotFoundError("Could not read images/sample.jpg")

# Use a flipped copy so both images have the same dimensions.
img2 = cv2.flip(img1, 1)

# alpha controls the contribution of the first image.
alpha = 0.5
beta = 1.0 - alpha
blended = cv2.addWeighted(img1, alpha, img2, beta, 0)

cv2.imwrite("output/blended_sample.jpg", blended)
print(f"Blended image saved. alpha={alpha}, beta={beta}")

cv2.imshow("Blended", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
