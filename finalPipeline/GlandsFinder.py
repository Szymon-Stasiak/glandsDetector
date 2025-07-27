from ultralytics import YOLO
import tifffile
import math
import random
from unet_core.unet_interface import UNET
import numpy as np
import cv2
from PIL import Image

def detect_on_tiles(tiles, positions, model):
    all_detections = []
    for tile, (x_offset, y_offset) in zip(tiles, positions):
        results = model(tile, conf=0.9)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 += x_offset
            x2 += x_offset
            y1 += y_offset
            y2 += y_offset
            conf = float(box.conf)
            cls = int(box.cls)
            all_detections.append((x1, y1, x2, y2, conf, cls))
    return all_detections


def merge_overlapping_boxes(
        detections,
        merge_distance=80,
        size_similarity_thresh=0.3
):
    merged = []
    used = [False] * len(detections)

    for i, det in enumerate(detections):
        if used[i]:
            continue

        x1, y1, x2, y2, conf, cls = det
        w1 = x2 - x1
        h1 = y2 - y1
        cx1 = (x1 + x2) / 2
        cy1 = (y1 + y2) / 2
        group = [(x1, y1, x2, y2, conf)]
        used[i] = True

        for j in range(i + 1, len(detections)):
            if used[j]:
                continue

            x1_b, y1_b, x2_b, y2_b, conf_b, cls_b = detections[j]
            w2 = x2_b - x1_b
            h2 = y2_b - y1_b
            cx2 = (x1_b + x2_b) / 2
            cy2 = (y1_b + y2_b) / 2

            dist = math.hypot(cx1 - cx2, cy1 - cy2)
            if dist > merge_distance:
                continue

            width_similar = abs(w1 - w2) / max(w1, w2) <= size_similarity_thresh
            height_similar = abs(h1 - h2) / max(h1, h2) <= size_similarity_thresh
            if not (width_similar or height_similar):
                continue

            group.append((x1_b, y1_b, x2_b, y2_b, conf_b))
            used[j] = True

        xs1, ys1, xs2, ys2, confs = zip(*group)
        merged_box = (
            min(xs1),
            min(ys1),
            max(xs2),
            max(ys2),
            sum(confs) / len(confs),
            cls
        )
        merged.append(merged_box)

    return merged


def draw_detections_varied_colors(image, detections, class_names, thickness=2, font_scale=0.5):
    annotated = image.copy()
    random.seed(42)

    for idx, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls_id = det

        color = tuple(int(c) for c in np.random.randint(0, 256, size=3))

        label = f"{class_names[int(cls_id)]} {conf:.2f}"

        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        text_y = int(y1) - 5 if y1 > 10 else int(y1) + 15
        cv2.putText(
            annotated,
            label,
            (int(x1), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness=1,
            lineType=cv2.LINE_AA
        )

    return annotated


def tile_image_with_detection(image, tile_size, overlap, model):
    tiles = []
    h, w = image.shape[:2]
    stride = tile_size - overlap

    ys = list(range(0, h, stride))
    xs = list(range(0, w, stride))

    n_tiles_y = len(ys)
    n_tiles_x = len(xs)

    detection_matrix = np.empty((n_tiles_y, n_tiles_x), dtype=object)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tile = image[y:y_end, x:x_end]
            tiles.append(tile)
            results = model(tile, conf=0.9)[0]
            list_of_boxes = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 += x
                x2 += x
                y1 += y
                y2 += y
                conf = float(box.conf)
                cls = int(box.cls)
                list_of_boxes.append((x1, y1, x2, y2, conf, cls))

            detection_matrix[i, j] = (x, y, list_of_boxes)

    return tiles, detection_matrix


def draw_detections_from_matrix(image, detection_matrix, class_names):
    annotated = image.copy()
    for i in range(detection_matrix.shape[0]):
        for j in range(detection_matrix.shape[1]):
            x, y, detections = detection_matrix[i, j]

            if detections is not None:
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{class_names[cls]} {conf:.2f}"
                    cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return annotated


def sort_boxes_by_confidence(detection_matrix):
    sorted_matrix = np.empty_like(detection_matrix, dtype=object)
    for i in range(detection_matrix.shape[0]):
        for j in range(detection_matrix.shape[1]):
            x, y, detections = detection_matrix[i, j]
            if detections is not None:
                sorted_detections = sorted(detections, key=lambda det: det[4], reverse=True)
                sorted_matrix[i, j] = (x, y, sorted_detections)
            else:
                sorted_matrix[i, j] = (x, y, None)
    return sorted_matrix


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    overlap_ratio = interArea / float(min(boxAArea, boxBArea)) if min(boxAArea, boxBArea) > 0 else 0
    return iou, overlap_ratio


def dimensions_similar(boxA, boxB, dim_thresh=0.3):
    widthA = boxA[2] - boxA[0]
    heightA = boxA[3] - boxA[1]
    widthB = boxB[2] - boxB[0]
    heightB = boxB[3] - boxB[1]

    width_similar = abs(widthA - widthB) / max(widthA, widthB) < dim_thresh
    height_similar = abs(heightA - heightB) / max(heightA, heightB) < dim_thresh

    return width_similar or height_similar


def assign_box_ids(grid, iou_thresh=0.4, overlap_thresh=0.60, dim_thresh=0.25):
    processed = []
    next_id = 1
    result = []
    total_detections = 0

    for row in grid:
        result_row = []
        for x_cell, y_cell, dets in row:
            if not dets:
                result_row.append((x_cell, y_cell, []))
                continue

            dets_sorted = sorted(dets, key=lambda b: b[4], reverse=True)
            dets_with_id = []
            for box in dets_sorted:
                total_detections += 1
                assigned = None
                for prev in processed:
                    iou, overlap = compute_iou(box, prev['box'])
                    if (iou > iou_thresh or overlap > overlap_thresh) and dimensions_similar(box, prev['box'],
                                                                                             dim_thresh):
                        assigned = prev['id']
                        break

                if assigned is None:
                    assigned = next_id
                    next_id += 1

                processed.append({'box': box, 'id': assigned})
                dets_with_id.append((*box, assigned))

            result_row.append((x_cell, y_cell, dets_with_id))
        result.append(result_row)

    print(f"Total detections: {total_detections}")
    print(f"Unique IDs assigned: {next_id - 1}")
    return result


def build_unique_detections_matrix(annotated_grid):
    unique_boxes = {}

    for row in annotated_grid:
        for x_cell, y_cell, dets in row:
            for box in dets:
                x1, y1, x2, y2, conf, cls, box_id = box
                if box_id not in unique_boxes:
                    unique_boxes[box_id] = [x1, y1, x2, y2, conf, cls]
                else:
                    prev = unique_boxes[box_id]
                    merged = [
                        min(prev[0], x1),
                        min(prev[1], y1),
                        max(prev[2], x2),
                        max(prev[3], y2),
                        max(prev[4], conf),
                        cls
                    ]
                    unique_boxes[box_id] = merged

    return [(*vals, box_id) for box_id, vals in unique_boxes.items()]


def draw_detections(image, detections, class_names):
    annotated = image.copy()
    for x1, y1, x2, y2, conf, cls, box_id in detections:
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{class_names[int(cls)]} {conf:.2f} ID:{box_id}"
        cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated


def segment_on_tiles(image, unique_detections, modelUNET):
    segmentations = []

    for detection in unique_detections:
        x1, y1, x2, y2 = map(int, detection[:4])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        tile = image[y1:y2, x1:x2]
        if tile.size == 0:
            continue

        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tile_pil = Image.fromarray(tile_rgb)

        local_mask = modelUNET.find_points(tile_pil)

        global_mask = [
            [(px + x1, py + y1) for (px, py) in contour]
            for contour in local_mask
        ]

        segmentations.append({
            'bbox': (x1, y1, x2, y2),
            'mask': global_mask,
            'id': detection[6] if len(detection) > 6 else None,
        })
        print(f"Segmented tile at ({x1}, {y1}) to ({x2}, {y2}); found {len(global_mask)} contour(s)")

    return segmentations


def apply_segmentations_to_image(image, segmentations, color=(0, 255, 0), thickness=2):
    img_with_segmentations = image.copy()

    for i, seg in enumerate(segmentations):
        mask = seg.get('mask', [])
        if not mask:
            print(f"Segment {i}: empty mask, skipping")
            continue

        if isinstance(mask[0], (list, tuple)) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in mask[0]):
            pts = mask[0]
        else:
            pts = mask

        if not pts:
            print(f"Segment {i}: unpacked mask is empty, skipping")
            continue

        first_point = pts[0]
        if isinstance(first_point, dict) and 'x' in first_point and 'y' in first_point:
            pts = [(point["x"], point["y"]) for point in pts]
        elif isinstance(first_point, (list, tuple)) and len(first_point) == 2:
            pts = [(int(x), int(y)) for x, y in pts]
        else:
            raise ValueError(f"Unknown mask point format for segment {i}: {pts}")

        if len(pts) == 1:
            cv2.circle(img_with_segmentations, pts[0], radius=5, color=color, thickness=-1)
        else:
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_with_segmentations, [pts], isClosed=True, color=color, thickness=thickness)

    return img_with_segmentations


detectionModel = YOLO("../model/saved_models/Glands_Finder_Augumented_Data_best.pt")
modelUNET = UNET(img_height=50, img_width=50)
modelUNET.set_model(in_channels=3, out_channels=1,
                    name="C:/Users/stszy/miniconda3/Lib/site-packages/unet_core/GlandsFinder")

image_path = "../preprocessedData_v1/tissue_regions/1M01/tissue_region_0.tiff"
tile_size = 2048
overlap = 1024

image = tifffile.imread(image_path)
if image.ndim == 3 and image.shape[2] > 3:
    image = image[:, :, :3]

if image.ndim == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
class_names = detectionModel.names

tiles, positions_matrix = tile_image_with_detection(image, tile_size, overlap, detectionModel)

print("Number of tiles:", len(tiles))
print("Positions matrix shape:", positions_matrix.shape)

annotated = draw_detections_from_matrix(image, positions_matrix, class_names)

print("Printing image with every detection on it...")
cv2.imwrite("everyDetection.png", annotated)

sorted_positions_matrix = sort_boxes_by_confidence(positions_matrix)
matrix_ready_for_segmentation = assign_box_ids(sorted_positions_matrix)
unique_detections = build_unique_detections_matrix(matrix_ready_for_segmentation)
annotated = draw_detections(image, unique_detections, class_names)

print("Printing image with merged detection on it...")
cv2.imwrite("mergedDetection.png", annotated)
segmentations = segment_on_tiles(image, unique_detections, modelUNET)
segmented_image = apply_segmentations_to_image(image, segmentations)
print("Printing segmented image...")
cv2.imwrite("segmented.png", segmented_image)
