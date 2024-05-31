from ultralytics import YOLO
import cv2
import argparse 
import supervision as sv

def parse_arguments() -> argparse.Namespace: #to set the resolution of the webcam
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280,720],
        nargs=2,
        type=int
    )
    args = parser.parse_args();
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    # cap = cv2.imread('pothole.webp');
    cap = cv2.VideoCapture('p.mp4'); #opening webcam

    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
    
    model = YOLO("best.pt") #creating model from the best weights

    box_annotator = sv.BoxAnnotator( #for boundary box around the potholes
        thickness=2,
        text_thickness=2,
        text_scale = 1
    )

    while True:
        ret, frame = cap.read();
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        frame = box_annotator.annotate(scene=frame, detections=detections)
        cv2.imshow("yolov8",frame); #showing the webcam frame
        
        if(cv2.waitKey(30) == 27): #27 represents the esc key. if user hits esc webcam goes off
            break;

if __name__ == "__main__":
    main()