import cv2
import time
from models_helpers.detection import Detection
from tracker import Tracker
import config
import argparse
import os
from typing import Tuple
from tqdm import tqdm


def args_parser():
    """ Parses command-line arguments for the Areial Tracking Application.
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Areial Tracking Application")
    parser.add_argument("-i", "--input_video_path", type=str, default="../test_videos/uav0000339_00001_v.mp4", help="Path to the input video")
    parser.add_argument("-d", "--detection_model_path", type=str, default="model/detection_yolo11n.onnx", help="Path to the ONNX detection model")
    parser.add_argument("-r", "--reid_model_path", type=str, default="model/reid_resnet18.onnx", help="Path to the ONNX ReID model")
    parser.add_argument("-o", "--output_save_path", type=str, default="outputs", help="Path to save the output video")
    parser.add_argument("-c", "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run the models on: 'cpu' or 'cuda'")

    return parser.parse_args()

def create_video_writer(output_folder_path: str,
                        frame_size: Tuple[int, int],
                        save_video_name: str) -> cv2.VideoWriter:
    """ Creates a video writer object for saving the output video.
    Args:
        output_folder_path (str): The directory where the output video will be saved.
        frame_size (Tuple[int, int]): The size of the video frames (width, height).
        save_video_name (str): The name of the output video file.
    Returns:
        cv2.VideoWriter: The video writer object for saving the output video.
    """
    os.makedirs(output_folder_path, exist_ok=True)
    save_video_path = os.path.join(output_folder_path, save_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_video_path, fourcc, 1, frame_size)
    return video_writer

def main():
    """ Main function to run the Areial Tracking Application. It initializes the detection and tracking models, processes the input video frame by frame, and saves the output video with tracking annotations."""
    
    print("*" * 70)
    print(f" Areial Tracking Application ".center(70, "*"))
    print("*" * 70, end="\n\n")

    args = args_parser()

    
    try:
        device = args.device

        detection_config = config.DetectionConfig(device=device)
        
        detection_model = Detection(args.detection_model_path, detection_config)
        
        print("1) Detection model loaded successfully...")
        
    except Exception as e:
        raise Exception(f"Error initializing detection model: {e}")
        exit()

    try:
        tracker_config = config.TrackerConfig()
        reid_config = config.REIDConfig(device=device)
        tracker = Tracker(reid_model_path=args.reid_model_path, tracker_config=tracker_config, reid_config=reid_config)
        print("2) Tracker Initialized Successfully...")
        
    except Exception as e:
        raise Exception(f"Error initializing tracker: {e}")
        exit()
    

    input_video_path = args.input_video_path
    input_video_name = input_video_path.split("/")[-1].split(".")[0]

    cap = cv2.VideoCapture(args.input_video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {input_video_path}")
        exit()
    
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_name = f"{input_video_name}_output.mp4"

    video_writer = create_video_writer(args.output_save_path, (video_width, video_height), output_video_name)
    
    print(f"3) Processing video: {input_video_path} ...")
    print("hint: Press 'q' to stop the video processing early.", end="\n\n")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Processing Video") as pbar:
        while cap.isOpened():
            start_time = time.perf_counter()

            ret, frame = cap.read()
            
            if not ret:
                break

            all_detections = detection_model(frame)

            if all_detections.size != 0:
                tracking_ids, boxes, tails = tracker.track(all_detections, frame)
                frame = tracker.draw(frame, tracking_ids, boxes, tails)

            end_time = time.perf_counter()
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 8)
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            video_writer.write(frame)
            pbar.update(1)
    
    cv2.destroyAllWindows()
    video_writer.release()
    print(f"4) Output video saved to: {os.path.join(args.output_save_path, output_video_name)} ...")
    print()
    print("*" * 70)
    print(f" Processing completed... ".center(70, "*"))
    print("*" * 70, end="\n\n")

if __name__ == "__main__":
    main()