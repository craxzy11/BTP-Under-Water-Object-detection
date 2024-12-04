from ultralytics import YOLO

# Load the trained model
model = YOLO("./runs/detect/train/weights/best.pt")  # Path to the trained model

# Test the model
results = model.val(
    data="./data.yaml",  # Path to the dataset configuration file
    split="test",  # Test split
    save=True,  # Save results
)

# Print out performance metrics
print("Test Results:")
print(f"mAP@0.5: {results['metrics/mAP50']:.4f}")
print(f"mAP@0.5:0.95: {results['metrics/mAP50-95']:.4f}")
print(f"Precision: {results['metrics/precision']:.4f}")
print(f"Recall: {results['metrics/recall']:.4f}")
