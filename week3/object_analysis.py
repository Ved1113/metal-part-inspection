import cv2

INPUT_IMAGE = "images/scattered_objects.jpg"
OUTPUT_IMAGE = "output/object_analysis.jpg"

image = cv2.imread(INPUT_IMAGE)

if image is None:
    raise FileNotFoundError(
        f"Could not read {INPUT_IMAGE}"
    )

# Convert BGR image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# -----------------------------
# Segment objects from background
# -----------------------------
# The image has a turquoise background.
# Create a mask for pixels that are NOT turquoise.

lower_bg = (75, 50, 50)
upper_bg = (105, 255, 255)

background_mask = cv2.inRange(
    hsv,
    lower_bg,
    upper_bg
)

# Invert mask:
# objects = white, background = black
binary = cv2.bitwise_not(background_mask)

# -----------------------------
# Clean the binary image
# -----------------------------
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel
)

# -----------------------------
# Find contours
# -----------------------------
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = image.copy()

object_count = 0

# -----------------------------
# Analyze objects
# -----------------------------
for contour in contours:

    area = cv2.contourArea(contour)

    # Ignore tiny noise
    if area < 300:
        continue

    object_count += 1

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)

    # Aspect ratio
    aspect_ratio = w / float(h)

    # Perimeter
    perimeter = cv2.arcLength(contour, True)

    # Circularity
    if perimeter > 0:
        circularity = (
            4 * 3.14159 * area
            / (perimeter * perimeter)
        )
    else:
        circularity = 0

    # Minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)

    center = (int(cx), int(cy))
    radius = int(radius)

    # -----------------------------
    # Size classification
    # -----------------------------
    if area < 1000:
        size = "Small"
    elif area < 2500:
        size = "Medium"
    else:
        size = "Large"

    # -----------------------------
    # Shape classification
    # -----------------------------
    if circularity > 0.75:
        shape = "Round"
    elif aspect_ratio < 0.75 or aspect_ratio > 1.33:
        shape = "Elongated"
    else:
        shape = "Compact"

    label = f"{size} {shape}"

    # Bounding box
    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Enclosing circle
    cv2.circle(
        result,
        center,
        radius,
        (255, 0, 0),
        2
    )

    # Label
    cv2.putText(
        result,
        f"{object_count}: {label}",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )

    print(
        f"Object {object_count}: "
        f"Area={area:.0f}, "
        f"Aspect Ratio={aspect_ratio:.2f}, "
        f"Circularity={circularity:.2f}, "
        f"Shape={shape}, "
        f"Size={size}"
    )

# -----------------------------
# Save results
# -----------------------------
cv2.imwrite(OUTPUT_IMAGE, result)

# Also save binary mask for verification
cv2.imwrite("output/object_mask.jpg", binary)

print()
print("Total objects detected:", object_count)
print("Object analysis completed.")
print(f"Saved as {OUTPUT_IMAGE}")
print("Saved as output/object_mask.jpg")