# YOLO-Based Human Pose Detection in Video

## Project Overview
This project uses YOLO (You Only Look Once) and Python to detect people in a video and visualize their body poses.

The system draws bounding boxes around detected people and identifies visible body keypoints, such as shoulders, elbows, wrists, hips, knees, and ankles when visible.

## Technologies Used
- Python
- YOLOv8 Pose Estimation
- OpenCV
- Ultralytics

## Project Files
- `code.py` – Python implementation of human pose detection.
- `input.mp4` – Input video.
- `output_pose.mp4` – Output video containing detection annotations.
- `requirements.txt` – Required Python libraries.

## Installation
Install the required libraries using:

```bash
pip install -r requirements.txt
```

## How to Run
1. Download or clone this repository.
2. Ensure `input.mp4` is in the project directory.
3. Run the Python script:

```bash
python code.py
```

The program processes the input video, displays the annotated frames, and saves the result as `output_pose.mp4`.

## Model
The project uses the `yolov8n-pose.pt` pretrained model provided by Ultralytics. The model may be downloaded automatically when the script runs for the first time.

## Output
The output video displays bounding boxes around detected people and pose keypoints for visible body parts.

## Author
Chahat Chaudhary
