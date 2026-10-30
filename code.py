import cv2
from ultralytics import YOLO

# Step 1: Load the YOLO pose estimation model
model = YOLO("yolov8n-pose.pt")

# Step 2: Specify the input and output video paths
input_video = "input.mp4"
output_video = "output_pose.mp4"

# Step 3: Open the input video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open the input video.")
    exit()

# Step 4: Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Handle videos with invalid FPS information
if fps <= 0:
    fps = 25.0

# Step 5: Create the output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("Error: Could not create the output video.")
    cap.release()
    exit()

print("Processing video... Press 'q' to stop.")

# Step 6: Process the video frame by frame
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Step 7: Detect people and estimate their poses
    results = model.predict(
        source=frame,
        conf=0.5,
        classes=[0],
        verbose=False
    )

    # Step 8: Draw bounding boxes and pose skeletons
    annotated_frame = results[0].plot()

    # Step 9: Display the processed frame
    cv2.imshow("YOLO Human Pose Detection", annotated_frame)

    # Step 10: Save the processed frame
    out.write(annotated_frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Step 11: Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed!")
print("Output saved as:", output_video)
