import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16


def main():
    # Load the trained SSD model
    model = ssd300_vgg16(weights=None)  # Sử dụng weights=None thay vì pretrained=False
    model.head.classification_head.num_classes = 2
    model.load_state_dict(torch.load('/Users/c/Downloads/ssd_strawberry_model (2 classes 70 epoch).pth',map_location=torch.device('cpu')))
    model.eval()
    
    # Define class labels
    class_labels = ['dau',"not dau"]


    # Mở camera
    cap = cv2.VideoCapture(0)
    transform = transforms.Compose([transforms.ToTensor()])

    while True:
        # Đọc khung hình từ camera
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame_resized = cv2.resize(frame, (300, 300))

        input_tensor = transform(frame_resized).unsqueeze(0)  # Thêm batch dimension
        with torch.no_grad():
            predictions = model(input_tensor)[0]  # Dự đoán bounding boxes
        
        # Vẽ bounding box lên frame
        
        if 'boxes' in predictions and 'scores' in predictions and 'labels' in predictions:
            for box, score,label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
                if ( 0.15 <= score <= 0.9 ) :  # Ngưỡng để loại bỏ các dự đoán ko có độ tin cậy
                    x1, y1, x2, y2 = map(int, box)
                    if label < len(class_labels):
                        
                        class_name = class_labels[label]
                        
                    else:
                        class_name = 'strawberry'
                        
                        
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (2, 255, 0), 2)
                    cv2.putText(frame_resized, f'{label}:{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Hiển thị khung hình
        cv2.imshow('Real-time Classification & Localization', frame_resized)
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
