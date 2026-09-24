import cv2

# Read image
img = cv2.imread("images/sample.jpg")

if img is None:
    print("Image not found")
    exit()

# Get image dimensions
height, width, channels = img.shape

print("Image shape:", img.shape)
print("Image height:", height)
print("Image width:", width)
print("Image channels:", channels)

# Calculate total number of pixels
total_pixel = height * width
print("Pixel count:", total_pixel)

# Select important pixel locations
locations = {
    "Top-Left": (0, 0),
    "Top-Right": (0, width - 1),
    "Bottom-Left": (height - 1, 0),
    "Bottom-Right": (height - 1, width - 1),
    "Center": (height // 2, width // 2)
}

# Display BGR pixel values
print("\nCOLOR IMAGE PIXEL VALUES:")
print("(OpenCV uses BGR order)")

for name, (y, x) in locations.items():
    pixel = img[y, x]

    print(
        f"{name}: "
        f"B={pixel[0]}, "
        f"G={pixel[1]}, "
        f"R={pixel[2]}"
    )

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display grayscale pixel values
print("\nGRAYSCALE PIXEL VALUES:")

for name, (y, x) in locations.items():
    b, g, r = img[y, x]

    gray_value = gray[y, x]

    average = (int(b) + int(g) + int(r)) / 3

    print(
        f"{name}: "
        f"Gray={gray_value}, "
        f"BGR Average={average:.2f}"
    )