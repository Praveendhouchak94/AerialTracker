from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DetectionConfig:

    device: str = "cpu" # "cpu" or "cuda"

    conf_thresh: float = 0.3
    iou_thresh: float = 0.65

    tile_size: int = 640
    overlap_horizontal: int = 30
    overlap_vertical: int = 60
    

    batch_size: int = 8

    def __post_init__(self):
        self.stride_horizontal = self.tile_size - self.overlap_horizontal
        self.stride_vertical = self.tile_size - self.overlap_vertical



@dataclass
class REIDConfig:

    device: str = "cpu" # "cpu" or "cuda"

    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    image_size: Tuple[int, int] = (128, 256)
    batch_size: int = 16



@dataclass
class TrackerConfig:
    track_high_thresh: float = 0.4
    track_low_thresh: float = 0.2
    new_track_thresh: float = 0.5
    match_thresh: float = 0.6
    track_buffer: int = 30
    frame_rate: int = 30
    mot20: bool = True
    gmc_method: str = "orb"
    proximity_thresh: float = 0.4
    appearance_thresh: float = 0.3
    with_reid: bool = True
    model: str = "auto"
    fuse_score: bool = True




    
