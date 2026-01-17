import ultralytics as ul
model = ul.YOLO(r"D:\Weed_detection_conference_paper\code\training\results_finetune\weed_split2\weights\best.pt")

def main():
    metrics = model.val(
        data = r"D:\Weed_detection_conference_paper\dataset\data.yaml"
    )
    print(metrics)
if __name__ == "__main__":
    main()