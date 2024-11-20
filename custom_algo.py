#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import copy
import time
import argparse

import cv2 as cv
from ultralytics import YOLO  # YOLO import
# print(cv.getBuildInformation())


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="sample_movie/bird.mp4")
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=360)

    # Existing tracker options
    parser.add_argument('--use_mil', action='store_true')
    parser.add_argument('--use_goturn', action='store_true')
    parser.add_argument('--use_dasiamrpn', action='store_true')
    parser.add_argument('--use_csrt', action='store_true')
    parser.add_argument('--use_kcf', action='store_true')
    parser.add_argument('--use_boosting', action='store_true')
    parser.add_argument('--use_mosse', action='store_true')
    parser.add_argument('--use_medianflow', action='store_true')
    parser.add_argument('--use_tld', action='store_true')
    parser.add_argument('--use_nano', action='store_true')
    parser.add_argument('--use_vit', action='store_true')

    # Add argument to enable YOLO detection
    parser.add_argument('--use_yolo', action='store_true', help='Use YOLO for object detection')

    args = parser.parse_args()

    return args

def isint(s):
    p = '[-+]?\d+'
    return True if re.fullmatch(p, s) else False

def create_tracker_by_name(tracker_algorithm):
    tracker = None
    if tracker_algorithm == 'MIL':
        tracker = cv.TrackerMIL_create()
    elif tracker_algorithm == 'GOTURN':
        params = cv.TrackerGOTURN_Params()
        params.modelTxt = "model/GOTURN/goturn.prototxt"
        params.modelBin = "model/GOTURN/goturn.caffemodel"
        tracker = cv.TrackerGOTURN_create(params)
    elif tracker_algorithm == 'DaSiamRPN':
        params = cv.TrackerDaSiamRPN_Params()
        params.model = "model/DaSiamRPN/dasiamrpn_model.onnx"
        params.kernel_r1 = "model/DaSiamRPN/dasiamrpn_kernel_r1.onnx"
        params.kernel_cls1 = "model/DaSiamRPN/dasiamrpn_kernel_cls1.onnx"
        tracker = cv.TrackerDaSiamRPN_create(params)
    elif tracker_algorithm == 'Nano':
        params = cv.TrackerNano_Params()
        params.backbone = "model/nanotrackv2/nanotrack_backbone_sim.onnx"
        params.neckhead = "model/nanotrackv2/nanotrack_head_sim.onnx"
        tracker = cv.TrackerNano_create(params)
    elif tracker_algorithm == 'Vit':
        params = cv.TrackerVit_Params()
        params.net = "model/vit/object_tracking_vittrack_2023sep.onnx"
        tracker = cv.TrackerVit_create(params)
    elif tracker_algorithm == 'CSRT':
        tracker = cv.TrackerCSRT_create()
    elif tracker_algorithm == 'KCF':
        tracker = cv.TrackerKCF_create()
    elif tracker_algorithm == 'Boosting':
        tracker = cv.legacy.TrackerBoosting_create()
    elif tracker_algorithm == 'MOSSE':
        tracker = cv.legacy.TrackerMOSSE_create()
    elif tracker_algorithm == 'MedianFlow':
        tracker = cv.legacy.TrackerMedianFlow_create()
    elif tracker_algorithm == 'TLD':
        tracker = cv.legacy.TrackerTLD_create()
    return tracker

def initialize_tracker_list(image, tracker_algorithm, bboxes):
    tracker_list = []
    for i, bbox in enumerate(bboxes):
        tracker = create_tracker_by_name(tracker_algorithm)
        if tracker is not None:
            tracker.init(image, bbox)
            tracker_list.append((tracker, tracker_algorithm))
    return tracker_list

def main():
    color_list = [
        [255, 0, 0],     # Blue
        [0, 255, 0],     # Green
        [0, 0, 255],     # Red
        [255, 255, 0],   # Cyan
        [255, 0, 255],   # Magenta
        [0, 255, 255],   # Yellow
        [128, 0, 128],   # Purple
        [128, 128, 0],   # Olive
        [0, 128, 128],   # Teal
        [128, 0, 0],     # Maroon
    ]

    # Parse arguments
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # Prepare tracker algorithm
    tracker_algorithm = None
    if args.use_mil:
        tracker_algorithm = 'MIL'
    elif args.use_goturn:
        tracker_algorithm = 'GOTURN'
    elif args.use_dasiamrpn:
        tracker_algorithm = 'DaSiamRPN'
    elif args.use_csrt:
        tracker_algorithm = 'CSRT'
    elif args.use_kcf:
        tracker_algorithm = 'KCF'
    elif args.use_boosting:
        tracker_algorithm = 'Boosting'
    elif args.use_mosse:
        tracker_algorithm = 'MOSSE'
    elif args.use_medianflow:
        tracker_algorithm = 'MedianFlow'
    elif args.use_tld:
        tracker_algorithm = 'TLD'
    elif args.use_nano:
        tracker_algorithm = 'Nano'
    elif args.use_vit:
        tracker_algorithm = 'Vit'

    # If no tracker is specified, default to CSRT
    if tracker_algorithm is None:
        tracker_algorithm = 'CSRT'

    use_yolo = args.use_yolo  # New argument for YOLO

    print("Tracker:", tracker_algorithm)
    print("Use YOLO:", use_yolo)

    # Open video capture
    if isint(cap_device):
        cap_device = int(cap_device)
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Initialize YOLO model if use_yolo is True
    if use_yolo:
        # yolo_model = YOLO('yolov8n.pt')
        yolo_model = YOLO("/home/artem-n/PycharmProjects/model/fly_last.pt") # You can choose a different model size

    # Initialize trackers
    window_name = 'Object Detection and Tracking'
    cv.namedWindow(window_name)

    ret, image = cap.read()
    if not ret:
        sys.exit("Can't read first frame")

    bboxes = []
    if use_yolo:
        # Use YOLO to detect objects and initialize trackers
        results = yolo_model.predict(image, stream_buffer=False)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                bboxes.append(bbox)
    else:
        # Manually select ROI
        bbox = cv.selectROI(window_name, image)
        bboxes.append(bbox)

    # Initialize tracker list
    tracker_list = initialize_tracker_list(image, tracker_algorithm, bboxes)

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        debug_image = image.copy()

        # Update trackers
        for index, (tracker, tracker_algorithm) in enumerate(tracker_list):
            ok, bbox = tracker.update(image)
            if ok:
                # Draw bounding box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                color = color_list[index % len(color_list)]
                cv.rectangle(debug_image, p1, p2, color, 2, 1)
                # Display tracker type on bounding box
                cv.putText(debug_image, f"{tracker_algorithm}", p1, cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            else:
                # Tracking failure
                cv.putText(debug_image, "Tracking failure detected", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv.imshow(window_name, debug_image)

        k = cv.waitKey(1)
        if k == 32:  # SPACE
            # Re-initialize trackers
            ret, image = cap.read()
            if not ret:
                break
            bboxes = []
            if use_yolo:
                # Re-detect objects using YOLO
                results = yolo_model(image)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                        bboxes.append(bbox)
            else:
                # Manually select ROI
                bbox = cv.selectROI(window_name, image)
                bboxes.append(bbox)
            tracker_list = initialize_tracker_list(image, tracker_algorithm, bboxes)
        elif k == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
