import cv2

img = cv2.imread("images/sample.jpg")

if img is None:
    print("Image not found!")
else:
    print("Image loaded successfully!")
    print("Shape:", img.shape)
    


    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
