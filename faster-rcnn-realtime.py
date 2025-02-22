import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

def main():
    # Load the trained Faster R-CNN model without COCO pre-trained weights
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    
    # Try to load the state dictionary
    try:
        model.load_state_dict(torch.load('/Users/c/Downloads/last_model_state.pth', map_location=torch.device('cpu')))
    except RuntimeError as e:
        print("Error loading state_dict:", e)
        model.load_state_dict(torch.load('/Users/c/Downloads/last_model_state.pth', map_location=torch.device('cpu')), strict=False)
    
    model.eval()

    # Move model to the right device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Open the default camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        frame = cv2.resize(frame, (300, 300))
        if not ret:
            break

        # Convert the frame to a tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.tensor(frame_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # Perform detection
        with torch.no_grad():
            predictions = model(frame_tensor)

        # Extract bounding boxes, labels, and scores
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        # Draw bounding boxes on the frame
        for box, score, label in zip(boxes, scores, labels):
            if 0.3<score < 0.6:  # Confidence threshold
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Score: {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Real-time Object Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# import cv2
# import onnxruntime as ort
# import numpy as np

# def main():
#     # Load ONNX model
#     onnx_model_path = "/Users/c/Downloads/fasterrcnn.onnx"
#     ort_session = ort.InferenceSession(onnx_model_path)

#     # Open the default camera
#     cap = cv2.VideoCapture(0)

#     while True:
#         # Read a frame from the camera
#         ret, frame = cap.read()
#         frame = cv2.resize(frame, (300, 300))
#         if not ret:
#             break

#         # Convert the frame to the required format
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_input = frame_rgb / 255.0  # Normalize pixel values to [0, 1]
#         frame_input = np.transpose(frame_input, (2, 0, 1))  # Change to channel-first format
#         frame_input = np.expand_dims(frame_input, axis=0).astype(np.float32)  # Add batch dimension

#         # Perform inference
#         inputs = {ort_session.get_inputs()[0].name: frame_input}
#         outputs = ort_session.run(None, inputs)

#         # Extract bounding boxes, labels, and scores
#         boxes = outputs[0]  # Assume 'boxes' is the first output
#         scores = outputs[1]  # Assume 'scores' is the second output
#         labels = outputs[2]  # Assume 'labels' is the third output

#         # Draw bounding boxes on the frame
#         for box, score, label in zip(boxes, scores, labels):
#             if 0.3 < score < 0.6:  # Confidence threshold
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f'Score: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Display the frame
#         cv2.imshow('Real-time Object Detection', frame)

#         # Press 'q' to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
