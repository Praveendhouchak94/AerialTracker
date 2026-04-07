import cv2
import numpy as np
import onnxruntime as ort
from config import DetectionConfig
from typing import List, Tuple, Union


class Detection:
    def __init__(self, model_path: str, config: Union[DetectionConfig, None] = None):
        """ Initializes the Detection class with the ONNX model and configuration parameters.
        
        Args:
            model_path (str): Path to the ONNX model file.
            detector_config (Union[DetectionConfig, None]): Configuration parameters for detection.    
            
        """
        
        self.config = config or DetectionConfig()

        
        self.__dict__.update(vars(self.config))

        
        self.conf_threshold = self.conf_thresh
        self.iou_threshold = self.iou_thresh


        self.conf_threshold = self.config.conf_thresh
        self.iou_threshold = self.config.iou_thresh
        self.tile_size = self.config.tile_size


        self.stride_horizontal = self.config.stride_horizontal
        self.stride_vertical = self.config.stride_vertical
        self.batch_size = self.config.batch_size
        self.device = self.config.device

        if self.device == "cuda":
            providers = ['CUDAExecutionProvider']
        elif self.device == "cpu":
            providers = ['CPUExecutionProvider']  
        else:
            raise ValueError(f"Unsupported device: {self.device}. Use 'cpu' or 'cuda'.")  
        
        self.model = ort.InferenceSession(model_path, providers=providers)  # use 'CUDAExecutionProvider' if GPU
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
    

    def get_tile_coords(self, w: int, h: int) -> List[Tuple[int, int]]:
        """ Computes top-left coordinates for tiling the image with specified overlap.
        Args:
            w (int): Width of the image.
            h (int): Height of the image.
        Returns:
        List[Tuple[int, int]]: List of (x, y) coordinates for the top-left corner of each tile.
            
        """
        xs = list(range(0, w - self.tile_size + 1, self.stride_horizontal))
        if xs[-1] != w - self.tile_size:
            xs.append(w - self.tile_size)

        ys = list(range(0, h - self.tile_size + 1, self.stride_vertical))
        if ys[-1] != h - self.tile_size:
            ys.append(h - self.tile_size)

        coords = []
        for y in ys:
            for x in xs:
                coords.append((x, y))
        return coords
    

 
    def preprocess_image(self, image: np.ndarray, 
                        coords: List[Tuple[int, int]]) -> List[np.ndarray]:
        """ Preprocesses image tiles in batches for ONNX model inference.
        
        Args:
            image (np.ndarray): Input image.
            coords (List[Tuple[int, int]]): List of tile coordinates.
        Returns:
            List[np.ndarray]: List of preprocessed batches ready for model input.
        """
        batches_inputs = []
        for i in range(0, len(coords), self.batch_size):
            batch_coords = coords[i: i+self.batch_size]
            tiles = []
            for (x, y) in batch_coords:
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                tile = cv2.resize(tile, (self.tile_size, self.tile_size))
                tile = tile[:, :, ::-1]  # BGR → RGB
                tile = tile.astype(np.float32) / 255.0
                tile = np.transpose(tile, (2, 0, 1))  # HWC → CHW
                tiles.append(tile)

            batch_input = np.stack(tiles, axis=0)
            batches_inputs.append(batch_input)
        return batches_inputs
    
    
    def inference(self, batch_input: List[np.ndarray]) -> np.ndarray:
        """
        Performs inference on a batch of preprocessed image tiles.
        
        Args:
            batch_input List[np.ndarray]: Batch of preprocessed image tiles.
        Returns:
            np.ndarray: Model outputs for the batch.
            """
        outputs = []
        for batch in batch_input:
            output = self.model.run([self.output_name], {self.input_name: batch})[0]
            outputs.append(output)
        return np.concatenate(outputs, axis=0)


    def post_processing_output(self, outputs: np.ndarray, batch_coords: List[Tuple[int, int]]) -> np.ndarray:
        """ Post-processes model outputs to extract bounding boxes, scores, and apply tile offsets.
        Args:
            outputs (np.ndarray): Raw model outputs for the batch.
            batch_coords (List[Tuple[int, int]]): List of tile coordinates corresponding to the batch.
        Returns:
            np.ndarray: Array of detections with global coordinates and scores."""
        preds_batch = outputs.transpose(0, 2, 1) 
        all_boxes = []
        all_scores = []
        batch_indices = []

        for b in range(preds_batch.shape[0]):
            preds = preds_batch[b]
            boxes = preds[:, :4]
            scores = preds[:, 4]
            mask = scores > self.conf_threshold
            if np.any(mask):
                all_boxes.append(boxes[mask])
                all_scores.append(scores[mask])
                batch_indices.extend([b] * np.sum(mask))

        if not all_boxes:
            return []

        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        batch_indices = np.array(batch_indices)

        # Vectorized offset addition
        tile_xs = np.array([batch_coords[b][0] for b in batch_indices])
        tile_ys = np.array([batch_coords[b][1] for b in batch_indices])

        centers_x = all_boxes[:, 0]
        centers_y = all_boxes[:, 1]
        widths = all_boxes[:, 2]
        heights = all_boxes[:, 3]

        x1 = centers_x - widths / 2 + tile_xs
        y1 = centers_y - heights / 2 + tile_ys
        x2 = centers_x + widths / 2 + tile_xs
        y2 = centers_y + heights / 2 + tile_ys  


        detections = np.column_stack([x1, y1, x2, y2, all_scores, np.zeros_like(all_scores)])  # (x1, y1, x2, y2, conf, class_id)
        return detections
    

    def global_nms(self, detections: np.ndarray) -> np.ndarray:
        """ 
        Applies Non-Maximum Suppression (NMS) to filter overlapping detections based on IoU threshold.
        
        Args:            
            detections (np.ndarray): Array of detections with format (x1, y1, x2, y2, score, class_id).
        Returns:            
            np.ndarray: Array of filtered detections after NMS.
        
        """

        if len(detections) == 0:
            return np.array([])

        # Split
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        width = detections[:, 2]
        height = detections[:, 3]
        x2 = x1 + width
        y2 = y1 + height
        
        scores = detections[:, 4]

        # Compute areas
        areas = (x2 - x1) * (y2 - y1)

        # Sort by score (descending)
        order = scores.argsort()[::-1]

        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)

            inter = w * h
            union = areas[i] + areas[order[1:]] - inter

            iou = inter / (union + 1e-6)

            # Keep boxes with IoU < threshold
            inds = np.where(iou < self.iou_threshold)[0]

            order = order[inds + 1]
        
        return detections[keep]
    

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Processes the input image through the entire detection pipeline: tiling, preprocessing, inference, post-processing, and NMS.
        Args:
            image (np.ndarray): Input image for detection.
        Returns:
            np.ndarray: Final array of detections after processing."""
        height, width, _ = image.shape
        coords = self.get_tile_coords(width, height)
        batches_inputs = self.preprocess_image(image, coords)
        model_outputs = self.inference(batches_inputs)
        detections = self.post_processing_output(model_outputs, coords)
        detections = self.global_nms(detections)
        return detections

        


    
