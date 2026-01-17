from ultralytics import YOLO

model = YOLO(r"D:\Weed_detection_conference_paper\code\training\yolov8n.pt")
def main():
    model.train(
        data="D:/Weed_detection_conference_paper/dataset/data.yaml",

        epochs=60,              # more than this = overfitting
        batch=8,                # small batch for small data
        imgsz=640,
        device=0,

        optimizer="AdamW",
        lr0=0.0005,             # LOW LR is critical
        lrf=0.01,
        cos_lr=True,

        patience=15,            # stop early if no improvement
        amp=True,

        # Augmentations (carefully tuned)
        mosaic=0.4,             # NOT 1.0 (too aggressive)
        mixup=0.0,              # turn off for small data
        scale=0.5,
        fliplr=0.5,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        project="results_small_data",
        name="weed_detector_157",
        exist_ok=True
    )
if __name__ == "__main__":
    main()