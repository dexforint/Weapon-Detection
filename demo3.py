import cv2
from ultralytics import YOLO
from time import time
from tools import *

print("Загрузка модели!")
model_name = "yolov8s_best.pt"
model = YOLO(model_name)
model.to("cuda")
print("Модель загружена!")

with open("stream_url.txt") as file:
    stream_url = file.read().strip()

cap = cv2.VideoCapture(stream_url)

time_amount = 0.0
counter = 0
shift = 25 * 35
index = -1
prev_frame = None

while cap.isOpened():
    index += 1
    success, frame = cap.read()
    if index < shift:
        continue

    if success:
        t1 = time()
        results = model(frame)[0]
        if not (prev_frame is None):
            crop, coords = get_motion_detection_field(frame, prev_frame)
            if not (crop is None or coords is None):
                local_results = model(crop)[0]
                # frame = cv2.rectangle(
                #     frame,
                #     (coords[0], coords[1]),
                #     (coords[0] + 640, coords[1] + 640),
                #     (0, 255, 0),
                #     2,
                # )
        else:
            local_results = None
            coords = None

        objects = unite_predictions(results, local_results, coords)

        t2 = time()
        time_amount += t2 - t1
        counter += 1
        annotated_frame = plot_objects(frame, objects)
        annotated_frame = cv2.resize(
            annotated_frame,
            (annotated_frame.shape[1] // 2, annotated_frame.shape[0] // 2),
        )

        cv2.imshow(model_name, annotated_frame)

        prev_frame = frame
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

print(time_amount / counter)

cap.release()
cv2.destroyAllWindows()
