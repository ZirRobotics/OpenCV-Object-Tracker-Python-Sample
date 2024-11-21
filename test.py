#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import copy
import time
import argparse

import cv2 as cv
import numpy as np
from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="sample_movie/bird.mp4")
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

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

    args = parser.parse_args()

    return args


def isint(s):
    p = '[-+]?\d+'
    return True if re.fullmatch(p, s) else False


def detect_objects(frame, model):
    """
    Object detection using YOLOv8.
    """
    # Perform detection
    results = model(frame)

    # Extract bounding boxes
    bboxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get the bounding box coordinates
            bboxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    return bboxes


def initialize_tracker_list(window_name, image, tracker_algorithm_list, detected_bboxes):
    tracker_list = []

    # Tracker list generation
    for tracker_algorithm in tracker_algorithm_list:
        for bbox in detected_bboxes:
            tracker = None
            if tracker_algorithm == 'MIL':
                tracker = cv.TrackerMIL_create()
            if tracker_algorithm == 'GOTURN':
                params = cv.TrackerGOTURN_Params()
                params.modelTxt = "model/GOTURN/goturn.prototxt"
                params.modelBin = "model/GOTURN/goturn.caffemodel"
                tracker = cv.TrackerGOTURN_create(params)
            if tracker_algorithm == 'DaSiamRPN':
                params = cv.TrackerDaSiamRPN_Params()
                params.model = "model/DaSiamRPN/dasiamrpn_model.onnx"
                params.kernel_r1 = "model/DaSiamRPN/dasiamrpn_kernel_r1.onnx"
                params.kernel_cls1 = "model/DaSiamRPN/dasiamrpn_kernel_cls1.onnx"
                tracker = cv.TrackerDaSiamRPN_create(params)
            if tracker_algorithm == 'Nano':
                params = cv.TrackerNano_Params()
                params.backbone = "model/nanotrackv2/nanotrack_backbone_sim.onnx"
                params.neckhead = "model/nanotrackv2/nanotrack_head_sim.onnx"
                tracker = cv.TrackerNano_create(params)
            if tracker_algorithm == 'Vit':
                params = cv.TrackerVit_Params()
                params.net = "model/vit/object_tracking_vittrack_2023sep.onnx"
                tracker = cv.TrackerVit_create(params)
            if tracker_algorithm == 'CSRT':
                tracker = cv.TrackerCSRT_create()
            if tracker_algorithm == 'KCF':
                tracker = cv.TrackerKCF_create()
            if tracker_algorithm == 'Boosting':
                tracker = cv.legacy_TrackerBoosting.create()
            if tracker_algorithm == 'MOSSE':
                tracker = cv.legacy_TrackerMOSSE.create()
            if tracker_algorithm == 'MedianFlow':
                tracker = cv.legacy_TrackerMedianFlow.create()
            if tracker_algorithm == 'TLD':
                tracker = cv.legacy_TrackerTLD.create()

            if tracker is not None:
                tracker.init(image, bbox)
                tracker_list.append(tracker)

    return tracker_list


def main():
    color_list = [
        [255, 0, 0],  # blue
        [255, 255, 0],  # aqua
        [0, 255, 0],  # lime
        [128, 0, 128],  # purple
        [0, 0, 255],  # red
        [255, 0, 255],  # fuchsia
        [0, 128, 0],  # green
        [128, 128, 0],  # teal
        [0, 0, 128],  # maroon
        [0, 128, 128],  # olive
        [0, 255, 255],  # yellow
    ]

    # Parse arguments ########################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_mil = args.use_mil
    use_goturn = args.use_goturn
    use_dasiamrpn = args.use_dasiamrpn
    use_csrt = args.use_csrt
    use_kcf = args.use_kcf
    use_boosting = args.use_boosting
    use_mosse = args.use_mosse
    use_medianflow = args.use_medianflow
    use_tld = args.use_tld
    use_nano = args.use_nano
    use_vit = args.use_vit

    # Tracker algorithm selection ############################################
    tracker_algorithm_list = []
    if use_mil:
        tracker_algorithm_list.append('MIL')
    if use_goturn:
        tracker_algorithm_list.append('GOTURN')
    if use_dasiamrpn:
        tracker_algorithm_list.append('DaSiamRPN')
    if use_csrt:
        tracker_algorithm_list.append('CSRT')
    if use_kcf:
        tracker_algorithm_list.append('KCF')
    if use_boosting:
        tracker_algorithm_list.append('Boosting')
    if use_mosse:
        tracker_algorithm_list.append('MOSSE')
    if use_medianflow:
        tracker_algorithm_list.append('MedianFlow')
    if use_tld:
        tracker_algorithm_list.append('TLD')
    if use_nano:
        tracker_algorithm_list.append('Nano')
    if use_vit:
        tracker_algorithm_list.append('Vit')

    if len(tracker_algorithm_list) == 0:
        tracker_algorithm_list.append('DaSiamRPN')
    print(tracker_algorithm_list)

    # Camera setup ###########################################################
    if isint(cap_device):
        cap_device = int(cap_device)
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Load YOLOv8 model ######################################################
    model = YOLO(r"D:\pycharm_projects\yolov8\runs\detect\drone_v9_300ep_32bath\weights\best.pt", task='detect')  # Ensure you have the correct path to your YOLOv8 model

    # Tracker initialization #################################################
    window_name = 'Tracker Demo'
    cv.namedWindow(window_name)

    tracker_list = []
    detected_bboxes = []

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        # If no tracker is initialized, run detection until an object is found
        if not tracker_list:
            detected_bboxes = detect_objects(image, model)
            if detected_bboxes:
                tracker_list = initialize_tracker_list(window_name, image, tracker_algorithm_list, detected_bboxes)

        elapsed_time_list = []
        tracker_scores = []  # Initialize a list to store tracker scores

        for index, tracker in enumerate(tracker_list):
            # Update tracking
            start_time = time.time()
            ok, bbox = tracker.update(image)
            try:
                tracker_score = tracker.getTrackingScore()
            except:
                tracker_score = '-'

            elapsed_time_list.append(time.time() - start_time)
            tracker_scores.append(tracker_score)  # Append the score to the list

            if ok:
                # Draw bounding box after tracking
                new_bbox = [
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3])
                ]
                cv.rectangle(debug_image,
                             (new_bbox[0], new_bbox[1]),
                             (new_bbox[0] + new_bbox[2], new_bbox[1] + new_bbox[3]),
                             color_list[index % len(color_list)],
                             thickness=2)
            else:
                # If tracking fails, reset trackers
                tracker_list = []
                break

        # Display processing time and tracker scores for each algorithm
        for index, tracker_algorithm in enumerate(tracker_algorithm_list):
            if index < len(elapsed_time_list):
                elapsed_time_ms = elapsed_time_list[index] * 1000
                if index < len(tracker_scores):
                    score = tracker_scores[index]
                    if score != '-':
                        text = f"{tracker_algorithm} : {elapsed_time_ms:.1f}ms Score:{score:.2f}"
                    else:
                        text = f"{tracker_algorithm} : {elapsed_time_ms:.1f}ms"
                else:
                    text = f"{tracker_algorithm} : {elapsed_time_ms:.1f}ms"
            else:
                text = f"{tracker_algorithm} : N/A"

            cv.putText(
                debug_image,
                text,
                (10, int(25 * (index + 1))),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                color_list[index % len(color_list)],
                2,
                cv.LINE_AA
            )

        cv.imshow(window_name, debug_image)

        k = cv.waitKey(1)
        if k == 32:  # SPACE
            # Reinitialize trackers based on new selection
            detected_bboxes = detect_objects(image, model)
            tracker_list = initialize_tracker_list(window_name, image, tracker_algorithm_list, detected_bboxes)
        if k == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
