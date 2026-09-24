import cv2

img = cv2.imread('images/group.webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(f'Faces found: {len(faces)}')

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    face_region_gray = gray[y:y+h, x:x+w]
    face_region_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_region_gray, scaleFactor=1.1, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face_region_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

cv2.imwrite('faces_detected.jpg', img)
print('Saved as faces_detected.jpg')