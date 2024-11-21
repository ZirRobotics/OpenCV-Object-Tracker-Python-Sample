#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import logging
import queue
import sys
import threading
import time

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


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Object Tracking Application")

    parser.add_argument("--device", default="sample_movie/bird.mp4",
                        help="Video source. Use integer for webcam or string for video file.")
    parser.add_argument("--width", type=int, default=1280,
                        help="Capture width")
    parser.add_argument("--height", type=int, default=720,
                        help="Capture height")

    parser.add_argument('--scale_factor', type=float, default=0.8,
                        help="Scale factor for downscaling frames for tracking")
    parser.add_argument('--large_object_threshold', type=int, default=500,
                        help="Threshold area in scaled image to decide tracker algorithm")

    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the YOLOv8 model")

    args = parser.parse_args()
    logging.debug(f"Parsed arguments: {args}")

    return args


def is_integer(s):
    """Check if the input string is an integer."""
    return s.isdigit() or (s.startswith('-') and s[1:].isdigit())


class ObjectDetector(threading.Thread):
    """Object detector running in a separate thread."""
    def __init__(self, model, frame_queue, detection_queue, stop_event):
        super().__init__(daemon=True)
        self.model = model
        self.frame_queue = frame_queue
        self.detection_queue = detection_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Reduced timeout to check stop_event more frequently
                frame = self.frame_queue.get(timeout=0.1)
                logging.debug("Frame retrieved from frame_queue for detection.")
                bboxes = self.detect_objects(frame)
                self.detection_queue.put(bboxes)
                logging.debug("Object detection completed and results put into detection_queue.")
            except queue.Empty:
                continue  # Loop again and check stop_event
            except Exception as e:
                logging.error(f"Exception in ObjectDetector: {e}")
                continue  # Continue the loop even if an error occurs

    def detect_objects(self, frame):
        """Perform object detection on the frame."""
        try:
            results = self.model(frame)
            bboxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    bboxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            logging.debug(f"Detected {len(bboxes)} bounding boxes.")
            return bboxes
        except Exception as e:
            logging.error(f"Error during object detection: {e}")
            return []


class TrackerFactory:
    """
    Factory class to handle tracker creation based on specified algorithms.
    """
    def __init__(self):
        self.nano_backbone_path = "model/nanotrackv2/nanotrack_backbone_sim.onnx"
        self.nano_neckhead_path = "model/nanotrackv2/nanotrack_head_sim.onnx"

    def create_tracker(self, image, bbox, algorithm):
        """
        Create and initialize a tracker based on the specified algorithm.

        Args:
            image (numpy.ndarray): The image used for initializing the tracker.
            bbox (tuple): The bounding box for the object (x, y, w, h).
            algorithm (str): The algorithm to use for the tracker (e.g., 'Nano', 'KCF').

        Returns:
            dict or None: A dictionary containing tracker info or None if initialization fails.
        """
        tracker = None
        logging.debug(f"Initializing tracker '{algorithm}' with bbox: {bbox}")

        if algorithm == 'Nano':
            params = cv.TrackerNano_Params()
            params.backbone = self.nano_backbone_path
            params.neckhead = self.nano_neckhead_path
            tracker = cv.TrackerNano_create(params)
        elif algorithm == 'KCF':
            tracker = cv.TrackerKCF_create()
        else:
            logging.warning(f"Unknown tracker algorithm: {algorithm}")
            return None

        if tracker is not None:
            try:
                tracker.init(image, bbox)
                logging.debug(f"Successfully initialized '{algorithm}' tracker with bbox: {bbox}")
                return {'tracker': tracker, 'algorithm': algorithm, 'bbox': bbox}
            except Exception as e:
                logging.error(f"Exception during tracker initialization for '{algorithm}' with bbox {bbox}: {e}")
        else:
            logging.error(f"Failed to initialize '{algorithm}' tracker with bbox: {bbox}")
        return None


class TrackerManager:
    """Manager for handling multiple trackers."""
    def __init__(self, scale_factor, large_object_threshold, tracker_factory):
        self.scale_factor = scale_factor
        self.large_object_threshold = large_object_threshold
        self.trackers = []
        self.tracker_factory = tracker_factory

    def initialize_trackers(self, scaled_image, detected_bboxes):
        """Initialize trackers based on detected bounding boxes."""
        self.trackers = []
        for bbox in detected_bboxes:
            scaled_bbox = (
                int(bbox[0] * self.scale_factor),
                int(bbox[1] * self.scale_factor),
                int(bbox[2] * self.scale_factor),
                int(bbox[3] * self.scale_factor)
            )
            area = scaled_bbox[2] * scaled_bbox[3]
            algorithm = 'KCF' if area > self.large_object_threshold else 'Nano'
            tracker_info = self.tracker_factory.create_tracker(scaled_image, scaled_bbox, algorithm)
            if tracker_info:
                self.trackers.append(tracker_info)
        logging.info(f"Total trackers initialized: {len(self.trackers)}")

    def update_trackers(self, scaled_image):
        """Update all trackers and collect results."""
        threads = []
        lower_threshold = self.large_object_threshold * 0.8
        upper_threshold = self.large_object_threshold * 1.2

        def update_single_tracker(tracker_info):
            idx = tracker_info['index']
            try:
                tracker = tracker_info['tracker']
                algorithm = tracker_info['algorithm']
                start_time = time.time()
                ok, bbox = tracker.update(scaled_image)
                elapsed_time = time.time() - start_time
                try:
                    tracker_score = tracker.getTrackingScore()
                except AttributeError:
                    tracker_score = '-'

                tracker_info.update({
                    'ok': ok,
                    'bbox': bbox,
                    'elapsed_time': elapsed_time,
                    'score': tracker_score
                })

                w, h = bbox[2], bbox[3]
                area = w * h

                if algorithm == 'KCF' and area <= lower_threshold:
                    logging.info(f"Switching tracker {idx} from KCF to Nano due to size decrease.")
                    self.switch_tracker(tracker_info, scaled_image, 'Nano')
                elif algorithm == 'Nano' and area >= upper_threshold:
                    logging.info(f"Switching tracker {idx} from Nano to KCF due to size increase.")
                    self.switch_tracker(tracker_info, scaled_image, 'KCF')
            except Exception as e:
                logging.error(f"Exception in tracker {idx} ({algorithm}): {e}")
                tracker_info['ok'] = False

        for idx, tracker_info in enumerate(self.trackers):
            tracker_info['index'] = idx
            t = threading.Thread(target=update_single_tracker, args=(tracker_info,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return self.trackers

    def switch_tracker(self, tracker_info, scaled_image, new_algorithm):
        """Switch the tracker algorithm for a given tracker_info."""
        idx = tracker_info['index']
        bbox = tracker_info['bbox']
        new_tracker_info = self.tracker_factory.create_tracker(scaled_image, bbox, new_algorithm)

        if new_tracker_info:
            tracker_info.update({
                'tracker': new_tracker_info['tracker'],
                'algorithm': new_algorithm,
            })
            logging.info(f"Tracker {idx} switched to {new_algorithm}.")
        else:
            logging.error(f"Failed to switch tracker {idx} to {new_algorithm}.")
            tracker_info['ok'] = False

    def reset_trackers(self):
        """Reset all trackers."""
        self.trackers = []


def setup_camera(device, width, height):
    """Set up the video capture device."""
    if is_integer(device):
        device = int(device)
    cap = cv.VideoCapture(device)
    if not cap.isOpened():
        logging.error(f"Cannot open video source: {device}")
        sys.exit(1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    logging.info(f"Video capture started on {device} with resolution {width}x{height}.")
    return cap


def load_model(model_path):
    """Load the YOLOv8 model."""
    try:
        model = YOLO(model_path, task='detect')
        logging.info(f"YOLOv8 model loaded from {model_path}.")
        return model
    except Exception as e:
        logging.error(f"Failed to load YOLOv8 model: {e}")
        sys.exit(1)


def draw_fps(frame, prev_time, prev_fps):
    # Calculate current time and frame duration
    current_time = time.perf_counter()
    frame_duration = current_time - prev_time

    # Calculate instantaneous FPS
    instantaneous_fps = 1.0 / frame_duration if frame_duration > 0 else 0

    # Apply exponential smoothing
    fps = 0.9 * prev_fps + 0.1 * instantaneous_fps

    # Update previous time
    prev_time = current_time

    # Prepare FPS text
    fps_text = f"FPS: {fps:.2f}"

    # Calculate text size for positioning
    text_size, _ = cv.getTextSize(fps_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = 30

    # Draw FPS text on the frame
    cv.putText(frame, fps_text, (text_x, text_y),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    return prev_time, fps


def process_tracking_results(trackers, debug_image, tracker_manager, color_list, scale_factor):
    """
    Process tracking results, draw bounding boxes, and handle tracking failures.

    Args:
        trackers (list): List of tracker information dictionaries.
        debug_image (numpy.ndarray): Image on which to draw.
        tracker_manager (TrackerManager): The tracker manager instance.
        color_list (list): List of colors for drawing.
        scale_factor (float): The scaling factor used for resizing images.
    """
    for tracker_info in trackers:
        index = tracker_info['index']
        ok = tracker_info['ok']
        bbox = tracker_info['bbox']
        algorithm = tracker_info['algorithm']
        elapsed_time = tracker_info['elapsed_time']
        score = tracker_info.get('score', '-')

        if ok:
            # Scale bbox back to original size
            x, y, w, h = bbox
            x = int(x / scale_factor)
            y = int(y / scale_factor)
            w = int(w / scale_factor)
            h = int(h / scale_factor)
            new_bbox = [x, y, w, h]

            # Draw bounding box
            cv.rectangle(debug_image,
                         (new_bbox[0], new_bbox[1]),
                         (new_bbox[0] + new_bbox[2], new_bbox[1] + new_bbox[3]),
                         color_list[index % len(color_list)],
                         thickness=2)
            logging.debug(f"Tracker {index} ({algorithm}) updated successfully with bbox: {new_bbox}")
        else:
            # If tracking fails, reset trackers
            logging.warning(f"Tracker {index} ({algorithm}) failed to update. Resetting trackers.")
            tracker_manager.reset_trackers()
            break

        # Display tracker info
        elapsed_time_ms = elapsed_time * 1000
        if score != '-':
            text = f"{algorithm} : {elapsed_time_ms:.1f}ms Score:{score:.2f}"
        else:
            text = f"{algorithm} : {elapsed_time_ms:.1f}ms"
        cv.putText(debug_image, text,
                   (10, int(25 * (index + 1))),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   color_list[index % len(color_list)],
                   2, cv.LINE_AA)


def main():
    # Parse arguments
    args = parse_arguments()

    # Initialize variables
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    scale_factor = args.scale_factor
    large_object_threshold = args.large_object_threshold
    model_path = args.model_path

    # Setup camera
    cap = setup_camera(cap_device, cap_width, cap_height)

    # Load YOLO model
    model = load_model(model_path)

    # Create queues and stop event
    frame_queue = queue.Queue(maxsize=1)
    detection_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    # Start detection thread
    detector = ObjectDetector(model, frame_queue, detection_queue, stop_event)
    detector.start()
    logging.info("Detection thread started.")

    tracker_factory = TrackerFactory()
    # Initialize tracker manager
    tracker_manager = TrackerManager(scale_factor, large_object_threshold, tracker_factory)

    # Initialize FPS variables
    prev_time = time.time()
    prev_fps = 0  # Initialize previous FPS for smoothing

    window_name = 'Tracker Demo'
    cv.namedWindow(window_name)

    color_list = [
        (255, 0, 0),     # blue
        (255, 255, 0),   # aqua
        (0, 255, 0),     # lime
        (128, 0, 128),   # purple
        (0, 0, 255),     # red
        (255, 0, 255),   # fuchsia
        (0, 128, 0),     # green
        (128, 128, 0),   # teal
        (0, 0, 128),     # maroon
        (0, 128, 128),   # olive
        (0, 255, 255),   # yellow
    ]

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.info("No frame received. Exiting main loop.")
                break

            # Calculate and display FPS using the draw_fps function
            prev_time, prev_fps = draw_fps(frame, prev_time, prev_fps)

            # debug_image = frame.copy()

            # Put frame into detection queue
            if not frame_queue.full():
                frame_queue.put(frame)
                logging.debug("Frame added to frame_queue.")
            else:
                logging.debug("Frame queue is full. Skipping frame.")

            # Retrieve detection results
            try:
                detected_bboxes = detection_queue.get_nowait()
                if detected_bboxes:
                    logging.debug(f"Retrieved {len(detected_bboxes)} bounding boxes from detection_queue.")
                    tracker_manager.initialize_trackers(frame, detected_bboxes)
            except queue.Empty:
                pass

            # Scale frame for tracking
            scaled_frame = cv.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)

            # Update trackers
            trackers = tracker_manager.update_trackers(scaled_frame)

            # Process tracking results using the new function
            process_tracking_results(trackers, frame, tracker_manager, color_list, scale_factor)

            # Display the frame
            cv.imshow(window_name, frame)

            # Handle key events
            k = cv.waitKey(1)
            if k == 27:  # ESC
                logging.info("ESC key pressed. Exiting.")
                break

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Exiting.")
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")
    finally:
        # Cleanup
        stop_event.set()
        detector.join(timeout=2)
        cap.release()
        cv.destroyAllWindows()
        logging.info("Resources released and program terminated gracefully.")


if __name__ == '__main__':
    main()
