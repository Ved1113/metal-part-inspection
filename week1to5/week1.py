import cv2
img =cv2.imread("images/sample.jpg")
if img is None:
    print("img not found")
    exit()

height ,width,channels=img.shape

print("img shape:",img.shape)
print("img height:",height)
print("img width:",width)
print("img channels:",channels)

Total_pixel =height*width
print("pixel count:",Total_pixel)

locations = {
    "top left" :(0,0),
    "Top-Right": (0, width - 1),
    "Bottom-Left": (height - 1, 0),
    "Bottom-Right": (height - 1, width - 1),
    "Center": (height // 2, width // 2)
}

print("\nCOLOR IMAGE PIXEL VALUES:")
print("(OpenCV uses BGR order)")

for name, (y, x) in locations.items():
    pixel = img[y, x]
    print(f"{name}: B={pixel[0]}, G={pixel[1]}, R={pixel[2]}")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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