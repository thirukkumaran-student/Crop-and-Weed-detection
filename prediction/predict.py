from ultralytics import YOLO
model = YOLO(r"D:\Weed_detection_conference_paper\code\training\results_small_data\weed_detector_157\weights\best.pt")

def main():
    result = model.predict(r"D:\Weed_detection_conference_paper\code\testing\input\Copy of IMG_20251225_163212.jpg")
    result[0].show()
if __name__ == "__main__":
    main()