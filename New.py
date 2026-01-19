import cv2
import torch
import numpy as np
import pyttsx3


engine = pyttsx3.init()
engine.setProperty('rate', 150)


model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


living_objects = ['person', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow']
non_living_objects = ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'traffic light', 'stop sign']


cap = cv2.VideoCapture(0)

prev_decision = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    frame_resized = cv2.resize(frame, (640, 480))


    with torch.no_grad():
        results = model(frame_resized, size=320)  # use smaller size for speed

    detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]
    labels = results.names

    decision = "Proceed"

    for *box, conf, cls in detections:
        class_name = labels[int(cls)]
        confidence = float(conf)

        if confidence > 0.5:
            if class_name in living_objects:
                decision = "Stop - Tanumaya sir  detected"
                break
            elif class_name in non_living_objects:
                decision = "Reroute - Non-Living Obstacle"


    if decision != prev_decision:
        engine.say(decision)
        engine.runAndWait()
        prev_decision = decision


    annotated_frame = np.squeeze(results.render())
    cv2.putText(annotated_frame,
                f"Decision: {decision}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255) if "Stop" in decision else (0, 255, 0),
                3)


    cv2.imshow("Autonomous Object Classification", annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
