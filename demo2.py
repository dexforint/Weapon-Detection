import cv2
from ultralytics import YOLO
from time import time
from tools import *

print("Загрузка модели!")
model_name = "yolov8s_best.pt"
model = YOLO(model_name)
model.to("cuda")
print("Модель загружена!")


with open("video_path.txt") as file:
    video_path = file.read().strip()
    video_name = ".".join(video_path.split("/")[-1].split(".")[:-1])

cap = cv2.VideoCapture(video_path)
width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = round(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

print(width, height, fps)

new_video = cv2.VideoWriter(
    f"{video_name}_processed.mp4",
    fourcc=fourcc,
    fps=fps,
    frameSize=(width, height),
)
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
        # annotated_frame = cv2.resize(
        #     annotated_frame,
        #     (annotated_frame.shape[1] // 2, annotated_frame.shape[0] // 2),
        # )

        prev_frame = frame
        new_video.write(annotated_frame)
    else:
        break

print(time_amount / counter)

cap.release()
new_video.release()
