#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import copy
import time
import argparse
import threading
import queue
import logging

import cv2 as cv
import numpy as np
from ultralytics import YOLO


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tracker_debug.log"),  # Log to file
        logging.StreamHandler(sys.stdout)          # Also log to console
    ]
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="sample_movie/bird.mp4")
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)

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
    logging.debug(f"Parsed arg: {args}")

    return args


def isint(s):
    p = '[-+]?\d+'
    logging.debug(f"Checking {p}")
    return True if re.fullmatch(p, s) else False


def detect_objects(model, frame, detection_queue):
    """
    Perform object detection and put the results in the detection_queue.
    """
    try:
        results = model(frame)
        bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Get the bounding box coordinates
                bboxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        detection_queue.put(bboxes)
        logging.debug(f"Detected {len(bboxes)} bounding boxes.")
    except Exception as e:
        logging.error(f"Error during object detection: {e}")


def initialize_tracker_list(image, tracker_algorithm_list, detected_bboxes):
    tracker_list = []

    # Tracker list generation
    for tracker_algorithm in tracker_algorithm_list:
        for bbox in detected_bboxes:
            tracker = None
            logging.debug(f"Initializing tracker '{tracker_algorithm}' with bbox: {bbox}")
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
                tracker = cv.legacy_TrackerBoosting.create()
            elif tracker_algorithm == 'MOSSE':
                tracker = cv.legacy_TrackerMOSSE.create()
            elif tracker_algorithm == 'MedianFlow':
                tracker = cv.legacy_TrackerMedianFlow.create()
            elif tracker_algorithm == 'TLD':
                tracker = cv.legacy_TrackerTLD.create()
            else:
                logging.warning(f"Unknown tracker algorithm: {tracker_algorithm}")

            if tracker is not None:
                try:
                    tracker.init(image, bbox)
                    tracker_list.append(tracker)
                    logging.debug(f"Successfully initialized '{tracker_algorithm}' tracker with bbox: {bbox}")
                except Exception as e:
                    logging.error(f"Exception during tracker initialization for '{tracker_algorithm}' with bbox {bbox}: {e}")
            else:
                logging.error(f"Failed to initialize '{tracker_algorithm}' tracker with bbox: {bbox}")

    logging.info(f"Total trackers initialized: {len(tracker_list)}")
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
    logging.info(f"Selected Tracker Algorithms: {tracker_algorithm_list}")
    print("Selected Tracker Algorithms:", tracker_algorithm_list)

    # Camera setup ###########################################################
    if isint(cap_device):
        cap_device = int(cap_device)
    cap = cv.VideoCapture(cap_device)
    if not cap.isOpened():
        logging.error(f"Cannot open video source: {cap_device}")
        sys.exit(1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    logging.info(f"Video capture started on {cap_device} with resolution {cap_width}x{cap_height}.")

    # Load YOLOv8 model ######################################################
    try:
        model_path = r"D:\pycharm_projects\yolov8\runs\detect\drone_v9_300ep_32bath\weights\best.pt"
        # model_path = "yolo11m.pt"
        model = YOLO(model_path, task='detect')  # Ensure you have the correct path to your YOLOv8 model
        logging.info(f"YOLOv8 model loaded from {model_path}.")
    except Exception as e:
        logging.error(f"Failed to load YOLOv8 model: {e}")
        sys.exit(1)

    # Queues for inter-thread communication ##################################
    frame_queue = queue.Queue(maxsize=1)
    detection_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    # Detection Thread ########################################################
    def detection_worker():
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1)
                logging.debug("Frame retrieved from frame_queue for detection.")
                detect_objects(model, frame, detection_queue)
                logging.debug("Object detection completed and results put into detection_queue.")
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Exception in detected_worker: {e}")

    detection_thread = threading.Thread(target=detection_worker, daemon=True)
    detection_thread.start()
    logging.info("Detection thread started.")

    # Tracker initialization #################################################
    window_name = 'Tracker Demo'
    cv.namedWindow(window_name)

    tracker_list = []
    detected_bboxes = []

    try:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                logging.info("No frame received. Exiting main loop.")
                break

            debug_image = copy.deepcopy(image)

            # Put the frame into the frame_queue for detection
            if not frame_queue.full():
                frame_queue.put(image)
                logging.debug("Frame added to frame_queue.")
            else:
                logging.debug("Frame queue is full. Skipping frame.")

            # Retrieve detection results if available
            try:
                detected_bboxes = detection_queue.get_nowait()
                if detected_bboxes:
                    logging.debug(f"Retrieved {len(detected_bboxes)} bounding boxes from detection_queue.")
                    tracker_list = initialize_tracker_list(image, tracker_algorithm_list, detected_bboxes)
                    logging.info(f"Initialized {len(tracker_list)} trackers based on detected bounding boxes.")
            except queue.Empty:
                pass  # No new detections yet

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

                elapsed_time = time.time() - start_time
                elapsed_time_list.append(elapsed_time)
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
                    logging.debug(f"Tracker {index} updated successfully with bbox: {new_bbox}")
                else:
                    # If tracking fails, reset trackers
                    logging.warning(f"Tracker {index} failed to update. Resetting all trackers.")
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
                detected_bboxes = detection_queue.get()
                tracker_list = initialize_tracker_list(image, tracker_algorithm_list, detected_bboxes)
            if k == 27:  # ESC
                logging.info("ESC key pressed. Exiting.")
                break
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Exiting.")
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")
    finally:
        # Cleanup ##############################################################
        stop_event.set()
        detection_thread.join(timeout=2)
        cap.release()
        cv.destroyAllWindows()
        logging.info("Resources released and program terminated gracefully.")


if __name__ == '__main__':
    main()
