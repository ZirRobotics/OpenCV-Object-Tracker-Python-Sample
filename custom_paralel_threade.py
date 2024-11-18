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
from collections import deque

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
    parser = argparse.ArgumentParser(description="Single Object Tracker with YOLOv8 Detection")

    parser.add_argument("--device", default="sample_movie/bird.mp4",
                        help="Video source. Can be a video file path or a camera index (integer).")
    parser.add_argument("--width", help='Capture width', type=int, default=960)
    parser.add_argument("--height", help='Capture height', type=int, default=540)

    # Single tracker selection using mutually exclusive group
    tracker_group = parser.add_mutually_exclusive_group()
    tracker_group.add_argument('--mil', action='store_true', help='Use MIL tracker')
    tracker_group.add_argument('--goturn', action='store_true', help='Use GOTURN tracker')
    tracker_group.add_argument('--dasiamrpn', action='store_true', help='Use DaSiamRPN tracker')
    tracker_group.add_argument('--nano', action='store_true', help='Use Nano tracker')
    tracker_group.add_argument('--vit', action='store_true', help='Use ViT tracker')
    tracker_group.add_argument('--csrt', action='store_true', help='Use CSRT tracker')
    tracker_group.add_argument('--kcf', action='store_true', help='Use KCF tracker')
    tracker_group.add_argument('--boosting', action='store_true', help='Use Boosting tracker')
    tracker_group.add_argument('--mosse', action='store_true', help='Use MOSSE tracker')
    tracker_group.add_argument('--medianflow', action='store_true', help='Use MedianFlow tracker')
    tracker_group.add_argument('--tld', action='store_true', help='Use TLD tracker')

    args = parser.parse_args()
    logging.debug(f"Parsed arguments: {args}")

    return args

def isint(s):
    p = '[-+]?\d+'
    logging.debug(f"Checking if '{s}' is an integer.")
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

class TrackerInfo:
    """
    A class to hold a single tracker and its bounding box history for smoothing.
    """
    def __init__(self, tracker, algorithm, initial_bbox, history_size=5):
        self.tracker = tracker
        self.algorithm = algorithm
        self.bbox_history = deque(maxlen=history_size)
        self.bbox_history.append(initial_bbox)
        self.color = None  # To be assigned later

    def update(self, frame):
        """
        Update the tracker with the current frame and store bbox.
        Returns the smoothed bbox and tracking status.
        """
        ok, bbox = self.tracker.update(frame)
        if ok:
            # Convert bbox to int and store
            bbox = [int(b) for b in bbox]
            self.bbox_history.append(bbox)
            # Calculate smoothed bbox
            smoothed_bbox = self.get_smoothed_bbox()
            return smoothed_bbox, True
        else:
            return None, False

    def get_smoothed_bbox(self):
        """
        Calculate the moving average of the bounding boxes.
        """
        avg_bbox = np.mean(self.bbox_history, axis=0)
        return [int(coord) for coord in avg_bbox]

def initialize_tracker(image, tracker_algorithm, initial_bbox):
    """
    Initialize a single tracker based on the specified algorithm and return TrackerInfo.
    """
    tracker = None
    logging.debug(f"Initializing tracker '{tracker_algorithm}'.")

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
        return None

    if tracker is not None:
        try:
            tracker.init(image, tuple(initial_bbox))
            tracker_info = TrackerInfo(tracker, tracker_algorithm, initial_bbox)
            logging.debug(f"Successfully initialized '{tracker_algorithm}' tracker with bbox: {initial_bbox}")
            return tracker_info
        except Exception as e:
            logging.error(f"Exception during tracker initialization for '{tracker_algorithm}' with bbox {initial_bbox}: {e}")
            return None
    else:
        logging.error(f"Failed to initialize '{tracker_algorithm}' tracker with bbox: {initial_bbox}")
        return None

def main():
    color_list = [
        [255, 0, 0],      # Blue
        [0, 255, 0],      # Green
        [0, 0, 255],      # Red
        [255, 255, 0],    # Cyan
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Yellow
        [128, 0, 128],    # Purple
        [128, 128, 0],    # Olive
        [0, 128, 128],    # Teal
        [128, 0, 0],      # Maroon
    ]

    # Parse arguments ########################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # Determine selected tracker algorithm
    tracker_algorithm = 'CSRT'  # Default tracker
    if args.mil:
        tracker_algorithm = 'MIL'
    elif args.goturn:
        tracker_algorithm = 'GOTURN'
    elif args.dasiamrpn:
        tracker_algorithm = 'DaSiamRPN'
    elif args.nano:
        tracker_algorithm = 'Nano'
    elif args.vit:
        tracker_algorithm = 'Vit'
    elif args.csrt:
        tracker_algorithm = 'CSRT'
    elif args.kcf:
        tracker_algorithm = 'KCF'
    elif args.boosting:
        tracker_algorithm = 'Boosting'
    elif args.mosse:
        tracker_algorithm = 'MOSSE'
    elif args.medianflow:
        tracker_algorithm = 'MedianFlow'
    elif args.tld:
        tracker_algorithm = 'TLD'

    logging.info(f"Selected Tracker Algorithm: {tracker_algorithm}")
    print("Selected Tracker Algorithm:", tracker_algorithm)

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
        model = YOLO(model_path, task='detect')  # Ensure you have the correct path to your YOLOv8 model
        logging.info(f"YOLOv8 model loaded from {model_path}.")
    except Exception as e:
        logging.error(f"Failed to load YOLOv8 model: {e}")
        sys.exit(1)

    # Queues for inter-thread communication ##################################
    frame_queue = queue.Queue(maxsize=5)  # Increased maxsize to 5
    detection_queue = queue.Queue(maxsize=5)
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
                logging.error(f"Exception in detection_worker: {e}")

    detection_thread = threading.Thread(target=detection_worker, daemon=True)
    detection_thread.start()
    logging.info("Detection thread started.")

    # Tracker initialization #################################################
    window_name = 'Tracker Demo'
    cv.namedWindow(window_name)

    tracker_info = None  # Single TrackerInfo instance
    detected_bboxes = []
    frame_count = 0
    detection_interval = 5  # Perform detection every 5 frames

    try:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                logging.info("No frame received. Exiting main loop.")
                break

            debug_image = copy.deepcopy(image)
            frame_count += 1

            # Put the frame into the frame_queue for detection every detection_interval frames
            if frame_count % detection_interval == 0:
                if frame_queue.full():
                    try:
                        removed_frame = frame_queue.get_nowait()
                        logging.debug("Frame queue is full. Removed oldest frame to add a new one.")
                    except queue.Empty:
                        logging.warning("Frame queue was full but no frame to remove.")
                frame_queue.put(image)
                logging.debug("Frame added to frame_queue.")
            else:
                logging.debug(f"Skipping detection for frame {frame_count}.")

            # Retrieve detection results if available
            try:
                detected_bboxes = detection_queue.get_nowait()
                if detected_bboxes and tracker_info is None:
                    logging.debug(f"Retrieved {len(detected_bboxes)} bounding boxes from detection_queue.")
                    # Initialize only the first detected bounding box
                    initial_bbox = detected_bboxes[0]
                    tracker_info = initialize_tracker(image, tracker_algorithm, initial_bbox)
                    if tracker_info:
                        # Assign a color
                        tracker_info.color = color_list[0 % len(color_list)]
                        logging.info(f"Initialized tracker '{tracker_algorithm}' with bbox: {initial_bbox}")
            except queue.Empty:
                pass  # No new detections yet

            if tracker_info:
                smoothed_bbox, ok = tracker_info.update(image)
                if ok and smoothed_bbox is not None:
                    # Draw bounding box after smoothing
                    cv.rectangle(debug_image,
                                 (smoothed_bbox[0], smoothed_bbox[1]),
                                 (smoothed_bbox[0] + smoothed_bbox[2], smoothed_bbox[1] + smoothed_bbox[3]),
                                 tracker_info.color,
                                 thickness=2)
                    logging.debug(f"Tracker ({tracker_info.algorithm}) updated successfully with smoothed bbox: {smoothed_bbox}")
                else:
                    # If tracking fails, remove the tracker_info
                    logging.warning(f"Tracker ({tracker_info.algorithm}) failed to update. Removing tracker.")
                    tracker_info = None

            # Display processing time and inference speed
            # Replace static text with dynamic measurements if desired
            # For now, keeping it static as in the original code
            cv.putText(
                debug_image,
                f"Speed: 1.0ms preprocess, 17.9ms inference, 11.0ms postprocess per image at shape (1, 3, 512, 640)",
                (10, debug_image.shape[0] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv.LINE_AA
            )

            cv.imshow(window_name, debug_image)

            k = cv.waitKey(1)
            if k == 32:  # SPACE
                # Reinitialize tracker based on new selection
                try:
                    new_detections = detection_queue.get_nowait()
                    if new_detections:
                        detected_bboxes = new_detections
                        # Reinitialize tracker with the first detected bounding box
                        initial_bbox = detected_bboxes[0]
                        tracker_info = initialize_tracker(image, tracker_algorithm, initial_bbox)
                        if tracker_info:
                            tracker_info.color = color_list[0 % len(color_list)]
                            logging.info(f"Re-initialized tracker '{tracker_algorithm}' with bbox: {initial_bbox}")
                except queue.Empty:
                    logging.debug("No new detections available for reinitialization.")
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
