import cv2

INPUT_PATH = "images/sample.jpg"

img = cv2.imread(INPUT_PATH)

if img is None:
    raise FileNotFoundError(f"Could not read image: {INPUT_PATH}")

# OpenCV reads color images in BGR order.
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Original BGR shape:", img.shape)
print("RGB shape:", rgb.shape)
print("HSV shape:", hsv.shape)
print("Gray shape:", gray.shape)

cv2.imwrite("output/rgb_sample.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite("output/hsv_sample.jpg", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
cv2.imwrite("output/gray_color_spaces.jpg", gray)

print("Saved RGB, HSV and grayscale outputs in output/")
