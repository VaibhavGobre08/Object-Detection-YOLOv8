import os
import random

import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
from ultralytics import YOLO

# Read labels from the file
with open("lables.txt", "r") as my_file:
    class_list = my_file.read().split("\n")


def speech(text):
    print(text)
    language = "en"
    output = gTTS(text=text, lang=language, slow=False)
    output.save("./sounds/output.mp3")
    # audio_file = os.path.dirname(__file__) + "\sounds\output.mp3"
    # print(audio_file)
    # playsound(audio_file)
    playsound("./sounds/output.mp3")


# Generate random colors for class list
detection_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(len(class_list))
]

model = YOLO("yolov8n.pt", "v8")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            detected_class = class_list[int(clsID)]
            print(
                f"Detected object: {detected_class}"
            )  # Print detected object in terminal
            string = str(f"Detected object: {detected_class}")
            speech(detected_class)

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                detected_class + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    cv2.imshow("ObjectDetection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
