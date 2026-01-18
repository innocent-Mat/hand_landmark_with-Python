import cv2
import mediapipe as mp
import numpy as npq
import uuid
import os

OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


mp_drawing = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
web_came = cv2.VideoCapture(0)

with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while web_came.isOpened():
        ret, frame = web_came.read()
        # convert to BGR2RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        # convert back to RGB2BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Landmarks
        # Face_landmarks
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hand.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=5),
                                          mp_drawing.DrawingSpec(color=(0, 200, 130), thickness=2, circle_radius=5))

        cv2.imwrite(os.path.join('output_images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow("Video", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

web_came.release()
cv2.destroyAllWindows()