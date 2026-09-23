import cv2

# Read the image
img = cv2.imread("images/sample.jpg")

# Check if image is loaded
if img is None:
    print("Error: Image not found. Check the file path.")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display images
    cv2.imshow("Original Image", img)
    cv2.imshow("Grayscale Image", gray)

    # Save grayscale image
    saved = cv2.imwrite("output/gray_sample.jpg", gray)

    # Check if image was saved
    if saved:
        print("✅ Image saved successfully in the 'output' folder.")
    else:
        print("❌ Failed to save the image.")

    # Wait for key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()