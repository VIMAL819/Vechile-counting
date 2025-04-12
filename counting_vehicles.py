import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLO model
model = YOLO("D:/Vehicle Detection/runs/detect/train2/weights/best.pt")

# Open the video file
video_path = "D:/Vehicle Detection/New folder/Test Video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Define the reference line position (lower in the frame)
line_y = int(frame_height * 0.75)
line_thickness = 2

# Vehicle tracking
vehicle_tracker = {}
vehicle_count = {"incoming": 0, "outgoing": 0}
next_vehicle_id = 0
tracked_objects = {}

# Function to calculate direction
def get_direction(start_y, end_y):
    if end_y > start_y:
        return "incoming"
    else:
        return "outgoing"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO model
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()
    
    new_objects = {}
    
    for det in detections:
        x1, y1, x2, y2, conf, _ = det  # Ignore class label
        if conf < 0.5:
            continue  # Filter out low-confidence detections
        
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Assign or track vehicle ID
        assigned_id = None
        min_dist = float('inf')
        
        for vid, (prev_x, prev_y) in tracked_objects.items():
            if prev_x != -1 and prev_y != -1:
                dist = np.sqrt((prev_x - center_x) ** 2 + (prev_y - center_y) ** 2)
                if dist < 50 and dist < min_dist:
                    assigned_id = vid
                    min_dist = dist
        
        if assigned_id is None:
            assigned_id = next_vehicle_id
            next_vehicle_id += 1
        
        new_objects[assigned_id] = (center_x, center_y)
        
        # Check if vehicle crosses the reference line
        prev_x, prev_y = tracked_objects.get(assigned_id, (-1, -1))
        if prev_y != -1:
            direction = get_direction(prev_y, center_y)
            
            if (prev_y < line_y <= center_y) or (prev_y > line_y >= center_y):
                if assigned_id not in vehicle_tracker:
                    vehicle_tracker[assigned_id] = direction
                    vehicle_count[direction] += 1
        
        # Draw bounding box and ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {assigned_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    tracked_objects = new_objects
    
    # Draw reference line
    cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 0, 255), line_thickness)
    
    # Display count
    cv2.putText(frame, f"Incoming: {vehicle_count['incoming']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Outgoing: {vehicle_count['outgoing']}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Write frame to output
    out.write(frame)
    
    # Display frame
    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()