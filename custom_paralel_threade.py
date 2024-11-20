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
from threading import Thread
import os  # Optional, for file path handling if needed

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
    p = r'[-+]?\d+'
    logging.debug(f"Checking if {s} is integer.")
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

def initialize_tracker_list(image, detected_bboxes, scale_factor, large_object_threshold):
    tracker_list = []

    # Scale the image for tracking
    scaled_image = cv.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)

    # Tracker list generation
    for bbox in detected_bboxes:
        # Scale the bounding box
        x, y, w, h = bbox
        scaled_bbox = (int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor))
        w_scaled, h_scaled = scaled_bbox[2], scaled_bbox[3]
        area = w_scaled * h_scaled

        # Decide which tracker to use based on the area
        if area > large_object_threshold:
            tracker_algorithm = 'KCF'
        else:
            tracker_algorithm = 'Nano'

        tracker = None
        logging.debug(f"Initializing tracker '{tracker_algorithm}' with bbox: {scaled_bbox}")

        if tracker_algorithm == 'Nano':
            params = cv.TrackerNano_Params()
            params.backbone = "model/nanotrackv2/nanotrack_backbone_sim.onnx"
            params.neckhead = "model/nanotrackv2/nanotrack_head_sim.onnx"
            tracker = cv.TrackerNano_create(params)
        elif tracker_algorithm == 'KCF':
            tracker = cv.TrackerKCF_create()
        else:
            logging.warning(f"Unknown tracker algorithm: {tracker_algorithm}")

        if tracker is not None:
            try:
                tracker.init(scaled_image, scaled_bbox)
                tracker_list.append((tracker, tracker_algorithm, scaled_bbox))
                logging.debug(f"Successfully initialized '{tracker_algorithm}' tracker with bbox: {scaled_bbox}")
            except Exception as e:
                logging.error(f"Exception during tracker initialization for '{tracker_algorithm}' with bbox {scaled_bbox}: {e}")
        else:
            logging.error(f"Failed to initialize '{tracker_algorithm}' tracker with bbox: {scaled_bbox}")

    logging.info(f"Total trackers initialized: {len(tracker_list)}")
    return tracker_list

def main():
    color_list = [
        [255, 0, 0],     # blue
        [255, 255, 0],   # aqua
        [0, 255, 0],     # lime
        [128, 0, 128],   # purple
        [0, 0, 255],     # red
        [255, 0, 255],   # fuchsia
        [0, 128, 0],     # green
        [128, 128, 0],   # teal
        [0, 0, 128],     # maroon
        [0, 128, 128],   # olive
        [0, 255, 255],   # yellow
    ]

    # Parse arguments ########################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # Set scale factor and large object threshold
    scale_factor = 0.8  # Downscale by half
    large_object_threshold = 5000  # Threshold area in scaled image

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
        # Adjust the path to your YOLOv8 model as needed
        model_path = r"D:\pycharm_projects\yolov8\runs\detect\drone_v9_300ep_32bath\weights\best.pt"
        model = YOLO(model_path, task='detect')
        logging.info(f"YOLOv8 model loaded from {model_path}.")
    except Exception as e:
        logging.error(f"Failed to load YOLOv8 model: {e}")
        sys.exit(1)

    # Video Writer setup #####################################################
    # output_video_path = 'output_video.mp4'  # Specify your desired output file path
    # fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Define the codec ('mp4v' for MP4 files)
    output_width = 1920
    output_height = 1080

    # Initialize the VideoWriter object
    # out = cv.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

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
                logging.error(f"Exception in detection_worker: {e}")

    detection_thread = threading.Thread(target=detection_worker, daemon=True)
    detection_thread.start()
    logging.info("Detection thread started.")

    # Tracker initialization #################################################
    window_name = 'Tracker Demo'
    cv.namedWindow(window_name)

    tracker_list = []
    detected_bboxes = []

    # Initialize FPS variables
    prev_time = time.time()
    fps = 0

    try:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                logging.info("No frame received. Exiting main loop.")
                break

            # Calculate FPS
            current_time = time.time()
            frame_duration = current_time - prev_time
            fps = 1.0 / frame_duration
            prev_time = current_time

            # Determine text position (top-right corner)
            fps_text = f"FPS: {fps:.2f}"
            text_size = cv.getTextSize(fps_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = image.shape[1] - text_size[0] - 10  # 10px padding from the right
            text_y = 30  # 30px from the top

            # Display FPS on the frame
            cv.putText(image, fps_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

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
                    tracker_list = initialize_tracker_list(image, detected_bboxes, scale_factor, large_object_threshold)
                    logging.info(f"Initialized {len(tracker_list)} trackers based on detected bounding boxes.")
            except queue.Empty:
                pass  # No new detections yet

            # Scale the image for tracking
            scaled_image = cv.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)

            # Initialize lists to store results
            elapsed_time_list = [None] * len(tracker_list)
            tracker_scores = [None] * len(tracker_list)
            ok_list = [None] * len(tracker_list)
            bbox_list = [None] * len(tracker_list)
            tracker_algorithms = [None] * len(tracker_list)

            # Function to update a single tracker
            def update_tracker(tracker_info, index, image):
                tracker, tracker_algorithm, _ = tracker_info
                start_time = time.time()
                ok, bbox = tracker.update(image)
                try:
                    tracker_score = tracker.getTrackingScore()
                except:
                    tracker_score = '-'
                elapsed_time = time.time() - start_time

                # Store results in the lists
                ok_list[index] = ok
                bbox_list[index] = bbox
                tracker_scores[index] = tracker_score
                elapsed_time_list[index] = elapsed_time
                tracker_algorithms[index] = tracker_algorithm

            # Start threads for each tracker
            threads = []
            for index, tracker_info in enumerate(tracker_list):
                t = Thread(target=update_tracker, args=(tracker_info, index, scaled_image))
                threads.append(t)
                t.start()

            # Wait for all threads to finish
            for t in threads:
                t.join()

            # Process the results
            for index in range(len(tracker_list)):
                ok = ok_list[index]
                bbox = bbox_list[index]
                tracker_score = tracker_scores[index]
                elapsed_time = elapsed_time_list[index]
                tracker_algorithm = tracker_algorithms[index]

                if ok:
                    # Scale bbox back to original size for display
                    x, y, w, h = bbox
                    x = int(x / scale_factor)
                    y = int(y / scale_factor)
                    w = int(w / scale_factor)
                    h = int(h / scale_factor)
                    new_bbox = [x, y, w, h]

                    # Draw bounding box after tracking
                    cv.rectangle(debug_image,
                                 (new_bbox[0], new_bbox[1]),
                                 (new_bbox[0] + new_bbox[2], new_bbox[1] + new_bbox[3]),
                                 color_list[index % len(color_list)],
                                 thickness=2)
                    logging.debug(f"Tracker {index} ({tracker_algorithm}) updated successfully with bbox: {new_bbox}")
                else:
                    # If tracking fails, reset trackers
                    logging.warning(f"Tracker {index} ({tracker_algorithm}) failed to update. Resetting all trackers.")
                    tracker_list = []
                    break

            # Display processing time and tracker scores for each tracker
            for index in range(len(tracker_list)):
                tracker_algorithm = tracker_algorithms[index]
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

            # Resize the debug_image to the desired output size
            debug_image_resized = cv.resize(debug_image, (output_width, output_height))

            # Write the resized frame to the output video
            # out.write(debug_image_resized)

            # Display the resized image
            cv.imshow(window_name, debug_image_resized)

            k = cv.waitKey(1)
            if k == 32:  # SPACE
                # Reinitialize trackers based on new selection
                detected_bboxes = detection_queue.get()
                tracker_list = initialize_tracker_list(image, detected_bboxes, scale_factor, large_object_threshold)
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
        # out.release()  # Release the VideoWriter
        cv.destroyAllWindows()
        logging.info("Resources released and program terminated gracefully.")

if __name__ == '__main__':
    main()
