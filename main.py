import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def blur_face(selected_image, nsize=9):
    rows, cols, _ = selected_image.shape
    dist = selected_image.copy()
    for y in range(0, rows, nsize):
        for x in range(0, cols, nsize):
            dist[y : y + nsize, x : x + nsize] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )


cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        res, img = cap.read()
        if not res:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writealbe to pass by peformance.
        img.flags.writeable = False
        results = face_detection.process(img)

        # Draw the face detection annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if results.detections:
            for detection in results.detections:
                bounding_box = detection.location_data.relative_bounding_box
                x = int(bounding_box.xmin * img.shape[1])
                y = int(bounding_box.ymin * img.shape[0])
                w = int(bounding_box.width * img.shape[1])
                h = int(bounding_box.height * img.shape[0])

                print(x, y, w, h)

                small = cv2.resize(
                    img[y : y + h, x : x + w],
                    None,
                    fx=0.05,
                    fy=0.05,
                    interpolation=cv2.INTER_NEAREST,
                )
                img[y : y + h, x : x + w] = cv2.resize(
                    small, (w, h), interpolation=cv2.INTER_NEAREST
                )

        cv2.imshow("Blur out Face", img)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
