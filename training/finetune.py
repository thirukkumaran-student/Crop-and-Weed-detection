from ultralytics import YOLO

model = YOLO(
    r"D:\Weed_detection_conference_paper\code\training\results\weed_detector3\weights\best.pt"  # from split-1
)
def main():
    model.train(
    data=r"D:\Weed_detection_conference_paper\dataset\data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,

    optimizer="AdamW",
    lr0=3e-4,          # LOWER LR for fine-tuning
    cos_lr=True,

    mosaic=0.4,
    mixup=0.0,

    amp=True,
    patience=10,

    project="results_finetune",
    name="weed_split2",
    exist_ok=True
)

if __name__ == "__main__":
    main()