import cv2

camera = cv2.VideoCapture(0)

frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

codec = cv2.VideoWriter_fourcc(*'XVID')

recorded = cv2.VideoWriter(
    "my_video.avi",
    codec,
    20,
    (frame_width, frame_height)
)

while True:
    success, image = camera.read()

    if not success:
        print("Could not read frame")
        break

    recorded.write(image)

    cv2.imshow("Recording Live", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
recorded.release()
cv2.destroyAllWindows()