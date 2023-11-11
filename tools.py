import numpy as np
from math import ceil
import cv2


def plot_objects(frame, objects):
    for cls, conf, box in objects:
        color = (255, 0, 0) if cls == 0 else (0, 0, 255)
        frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

    return frame


def non_max_supression(objects):
    if len(objects) == 0:
        return []

    filtered_objects = [objects.pop()]
    while len(objects) != 0:
        obj = objects.pop()
        for filtered_obj in filtered_objects:
            iou = get_iou(obj[2], filtered_obj[2])
            if iou > 0.6:
                break
        else:
            filtered_objects.append(obj)

    return filtered_objects


def unite_predictions(global_predictions, local_predictions, coords):
    classes = [int(el) for el in global_predictions.boxes.cls.tolist()]
    confs = global_predictions.boxes.conf.tolist()
    boxes = global_predictions.boxes.xyxy.tolist()

    if not (local_predictions is None or coords is None):
        x, y = coords

        classes.extend([int(el) for el in local_predictions.boxes.cls.tolist()])
        confs.extend(local_predictions.boxes.conf.tolist())

        local_boxes = local_predictions.boxes.xyxy.tolist()
        for i, local_box in enumerate(local_boxes):
            local_box = [
                local_box[0] + x,
                local_box[1] + y,
                local_box[2] + x,
                local_box[3] + y,
            ]
            boxes.append(local_box)

    boxes = [[round(el) for el in box] for box in boxes]

    objects = list(zip(classes, confs, boxes))
    objects.sort(key=lambda el: -el[1])

    weapons = [obj for obj in objects if obj[0] == 1]
    people = [obj for obj in objects if obj[0] == 0]

    weapons = non_max_supression(weapons)
    people = non_max_supression(people)

    return people + weapons


def get_motion_detection_field(frame, prev_frame):
    original_frame = frame
    H, W, _ = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (11, 11), 0)

    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.GaussianBlur(prev_frame, (11, 11), 0)

    frameDelta = cv2.absdiff(frame, prev_frame)
    thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, None

    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    boxes.sort(key=lambda el: -el[2] * el[3])
    boxes = boxes[:3]

    x1, y1, w, h = boxes[0]
    x = round(x1 + w / 2)
    y = round(y1 + h / 2)

    x1 = x - 320
    x2 = x + 320
    y1 = y - 320
    y2 = y + 320

    if x1 < 0:
        x1 = 0
        x2 = 640
    elif x2 >= W:
        x2 = W
        x1 = W - 640

    if y1 < 0:
        y1 = 0
        y2 = 640
    elif y2 >= H:
        y2 = H
        y1 = H - 640

    return original_frame[y1:y2, x1:x2], (x1, y1)


def yolo_output2predictions(pred):
    boxes = pred.boxes.xyxy.cpu().numpy()
    confs = pred.boxes.conf.cpu().numpy()

    result = {
        "boxes": boxes,
        "scores": confs,
        "labels": np.zeros((len(boxes),)),
        "boxes_int": boxes.astype(np.int32),
    }
    return result


def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = max(interArea / boxAArea, interArea / boxBArea)
    return iou


def nms(objects, threshold=0.5):
    if len(objects) == 0:
        return []

    objects.sort(key=lambda obj: obj[1])

    filtered_objects = [objects.pop()]
    while len(objects) != 0:
        obj = objects.pop()
        for filtered_obj in filtered_objects:
            iou = get_iou(obj[0], filtered_obj[0])
            if iou > threshold:
                break
        else:
            filtered_objects.append(obj)

    return filtered_objects


def filter_by_size(objects, thresholds):
    top_object = objects[0]
    conf = top_object[1]
    if conf < 0.5:
        return objects

    top_box = top_object[0]
    top_area = (top_box[2] - top_box[0]) * (top_box[3] - top_box[1])

    filtered_objects = []
    for obj in objects:
        box = obj[0]
        area = (box[2] - box[0]) * (box[3] - box[1])
        ratio = area / top_area

        if not (thresholds[0] is None) and not (thresholds[1] is None):
            if ratio > thresholds[0] and ratio < thresholds[1]:
                filtered_objects.append(obj)
        elif not (thresholds[0] is None):
            if ratio > thresholds[0]:
                filtered_objects.append(obj)
        elif not (thresholds[1] is None):
            if ratio < thresholds[1]:
                filtered_objects.append(obj)
        else:
            filtered_objects.append(obj)

    return filtered_objects


def postprocess(pred, nms_threshold=0.5, size_thresholds=(0.2, 3)):
    boxes = pred["boxes"].tolist()
    confs = pred["scores"].tolist()
    objects = list(zip(boxes, confs))

    objects = nms(objects, threshold=nms_threshold)
    if len(objects) > 0:
        boxes, confs = list(zip(*objects))
    else:
        boxes = []
        confs = []

    pred = {
        "boxes": [[round(val) for val in box] for box in boxes],
        "scores": confs,
    }

    return pred


def get_patches(img, patch_shape=(640, 640), step=(540, 540)):
    """Функция разрезает изображение на подизображения (патчи)"""
    H, W, _ = img.shape

    horizonatal_steps_num = ceil((W - patch_shape[0]) / step[0]) + 1
    vertical_steps_num = ceil((H - patch_shape[1]) / step[1]) + 1

    patches = []
    coords = []
    for i in range(horizonatal_steps_num):
        x_left = min(W - patch_shape[0], i * step[0])
        for j in range(vertical_steps_num):
            y_top = min(H - patch_shape[1], j * step[1])
            patch = img[
                y_top : y_top + patch_shape[1], x_left : x_left + patch_shape[0]
            ]
            patches.append(patch)
            coords.append((x_left, y_top))

    return patches, coords


def union_predictions(preds, coords):
    all_boxes = []
    all_scores = []

    for pred, (x_left, y_top) in zip(preds, coords):
        pred = yolo_output2predictions(pred)
        boxes = pred["boxes"]
        boxes[:, (0, 2)] += x_left
        boxes[:, (1, 3)] += y_top

        scores = pred["scores"]

        all_boxes.append(boxes)
        all_scores.append(scores)

    all_scores = np.concatenate(all_scores)
    all_boxes = np.concatenate(all_boxes)

    result = {
        "boxes": all_boxes,
        "scores": all_scores,
        "labels": np.zeros((len(all_boxes),)),
        "boxes_int": all_boxes.astype(np.int32),
    }
    return result
