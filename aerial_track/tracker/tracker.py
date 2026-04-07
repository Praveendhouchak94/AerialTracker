from collections import defaultdict, deque
import numpy as np
import cv2
from types import SimpleNamespace
from ultralytics.trackers.bot_sort import BOTSORT
from models_helpers.reid import ReIDModel
from ultralytics.engine.results import Boxes
from config import TrackerConfig, REIDConfig
from typing import Dict, List, Tuple, Union


class Tracker:
    def __init__(self, reid_model_path: str, tracker_config: Union[TrackerConfig, None] = None, reid_config: Union[REIDConfig, None] = None):
        """ Initializes the Tracker class with the ReID model and tracking configuration parameters.
        Args:
            reid_model_path (str): Path to the ONNX ReID model file.
            tracker_config (Union[TrackerConfig, None]): Configuration parameters for the tracker.
            reid_config (Union[REIDConfig, None]): Configuration parameters for the ReID model.
        """

        self.tracker_config = tracker_config or TrackerConfig()
        self.__dict__.update(vars(self.tracker_config))
        self.tracker = BOTSORT(SimpleNamespace(**vars(self.tracker_config)))

        self.reid_config = reid_config or REIDConfig()
        self.__dict__.update(vars(self.reid_config))

        self.reid_model_path = reid_model_path

        self.tracker.encoder = ReIDModel(self.reid_model_path, reid_config)

        self.track_history = defaultdict(lambda: deque(maxlen=30))


    @staticmethod
    def get_color(track_id: int) -> Tuple[int, int, int]:
        """ Generates a consistent color for each track ID using a seeded random generator.
        Args:
            track_id (int): The unique identifier for the track.
        Returns:
            Tuple[int, int, int]: The RGB color values for the track.
        """
        np.random.seed(track_id)
        return (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))


    def track(self, detections: np.ndarray,
               image: np.ndarray) -> Tuple[List[int], 
                                           List[Tuple[int, int, int, int]], 
                                           Dict[int, List[Tuple[int, int]]]]:
        """ Performs tracking by associating detections with existing tracks and updating the track history.
        Args:
            detections (np.ndarray): Array of detections with format (x1, y1, x2, y2, score, class_id).
            image (np.ndarray): The current video frame for visualization and ReID feature extraction.
        Returns:
            Tuple[List[int], List[Tuple[int, int, int, int]], Dict[int, List[Tuple[int, int]]]]: A tuple containing the list of tracking IDs, the list of bounding boxes, and the dictionary of track tails.
        """
        height, width = image.shape[:2]

        deletctions = Boxes(np.array(detections), [height, width])
        tracks = self.tracker.update(deletctions, image)
        
        tracking_ids = []
        boxes = []
        tails = {}

        active_ids = set()


        for track in tracks:
            x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
            tracking_ids.append(track_id)
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            active_ids.add(track_id)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # store history
            self.track_history[track_id].append((cx, cy))

            tails[track_id] = list(self.track_history[track_id])


        for tid in list(self.track_history.keys()):
            if tid not in active_ids:
                del self.track_history[tid]

        return tracking_ids, boxes, tails

    @staticmethod
    def dymanic_segmnet_pts(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2

        
        start1 = (x1, y1)
        end1 = (
            int(x1 + (x2 - x1) / 3),
            int(y1 + (y2 - y1) / 3)
        )

        start2 = (
            int(x1 + 2 * (x2 - x1) / 3),
            int(y1 + 2 * (y2 - y1) / 3)
        )
        end2 = (x2, y2)
        return [(start1, end1), (start2, end2)]

    def draw(self, frame: np.ndarray,
                tracking_ids: List[int],
                boxes: List[Tuple[int, int, int, int]],
                tails: Dict[int, List[Tuple[int, int]]]) -> np.ndarray:
        """ Draws the tracking results on the video frame, including bounding boxes, track IDs, and tails.
        Args:
            frame (np.ndarray): The video frame on which to draw the tracking results.
            tracking_ids (List[int]): List of tracking IDs for the current frame.
            boxes (List[Tuple[int, int, int, int]]): List of bounding boxes corresponding to the tracking IDs.
            tails (Dict[int, List[Tuple[int, int]]]): Dictionary containing the tail points for each track ID.
        Returns:
            np.ndarray: The video frame with the tracking results drawn on it.
        """

        for track_id, box in zip(tracking_ids, boxes):
            x1, y1, x2, y2 = map(int, box)
            lines = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
            
            color = self.get_color(int(track_id))

            for i in range(4):
                segments = self.dymanic_segmnet_pts(lines[i], lines[i + 1])
                for pt1, pt2 in segments:
                    cv2.line(frame, pt1, pt2, color, 2)

            cv2.putText(frame, f"ID {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2)

            # tail
            points = tails.get(track_id, [])
            for i in range(1, len(points)):
                thickness = int(2 * (i / len(points))) + 1
                cv2.line(frame, points[i - 1], points[i], color, thickness)

        return frame