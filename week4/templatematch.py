import cv2

# Read the main image and template image
image = cv2.imread("images/parts_scene.png")
template = cv2.imread("images/nut_template.png")

if image is None:
    raise FileNotFoundError("Could not read images/parts_scene.png")

if template is None:
    raise FileNotFoundError("Could not read images/nut_template.png")

# Get template dimensions
h, w = template.shape[:2]

# Perform template matching
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# Find the best match
_, max_val, _, max_loc = cv2.minMaxLoc(result)

# Draw rectangle around the best match
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

output = image.copy()

cv2.rectangle(
    output,
    top_left,
    bottom_right,
    (0, 255, 0),
    2
)

cv2.putText(
    output,
    f"Match Score: {max_val:.2f}",
    (top_left[0], max(top_left[1] - 10, 20)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

# Save result
cv2.imwrite("output/template_matching_result.jpg", output)

print("Template matching completed.")
print(f"Best match score: {max_val:.2f}")
print("Saved as output/template_matching_result.jpg")