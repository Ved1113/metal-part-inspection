import cv2

# Read input image
img = cv2.imread("images/group.webp")

if img is None:
    raise FileNotFoundError("Could not load images/group.webp")

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade models
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_alt2.xml"
)

eye_cascade = cv2.CascadeClassifier(
    "haarcascade_eye.xml"
)

if face_cascade.empty():
    raise FileNotFoundError(
        "Could not load haarcascade_frontalface_alt2.xml"
    )

if eye_cascade.empty():
    raise FileNotFoundError(
        "Could not load haarcascade_eye.xml"
    )

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
)

print(f"Faces found: {len(faces)}")

# Detect faces and eyes
for (x, y, w, h) in faces:

    # Draw face rectangle
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Extract face region
    face_region_gray = gray[y:y + h, x:x + w]
    face_region_color = img[y:y + h, x:x + w]

    # Detect eyes inside face
    eyes = eye_cascade.detectMultiScale(
        face_region_gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (ex, ey, ew, eh) in eyes:

        # Draw eye rectangle
        cv2.rectangle(
            face_region_color,
            (ex, ey),
            (ex + ew, ey + eh),
            (255, 0, 0),
            2
        )

# Save output
cv2.imwrite(
    "output/faces_detected.jpg",
    img
)

print("Saved as output/faces_detected.jpg")