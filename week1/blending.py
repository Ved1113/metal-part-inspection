import cv2

img1 = cv2.imread("images/sample.jpg")
img2 = cv2.imread("images/sample.jpg")

if img1 is None:
    raise FileNotFoundError("Could not read images/sample.jpg")

# Resize second image to match first image
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Blend the images
blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

cv2.imwrite("output/blended.jpg", blended)

print("Blending completed.")
print("Saved as output/blended.jpg")