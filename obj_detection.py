
# 1. Import libraries
import time
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch
import torchvision
from torchvision.transforms import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("Device:", device)


# 2. Load image
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_FILE = Path("object_detection_sample.jpg")

if not SAMPLE_FILE.exists():
    urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_FILE)

image = Image.open(SAMPLE_FILE).convert("RGB")

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.show()


# 3. Define visualization functions
def show_comparison(original, detected, title):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(detected)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def draw_detections(image, boxes, labels, scores,
                    categories, threshold=0.60):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    count = 0

    for box, label, score in zip(boxes, labels, scores):
        score = float(score)

        if score < threshold:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        class_id = int(label)
        name = categories[class_id]
        text = f"{name}: {score:.2f}"

        draw.rectangle(
            [x1, y1, x2, y2], outline="red", width=3
        )
        draw.text((x1 + 2, y1 + 2), text, fill="white")
        count += 1

    return output, count


# 4. Load Faster R-CNN
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights
)

frcnn_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT

frcnn = fasterrcnn_resnet50_fpn_v2(
    weights=frcnn_weights
).to(device).eval()

frcnn_categories = frcnn_weights.meta["categories"]
frcnn_preprocess = frcnn_weights.transforms()

print("Faster R-CNN loaded.")


# 5. Detect objects using Faster R-CNN
input_image = frcnn_preprocess(image).to(device)

if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

with torch.no_grad():
    frcnn_pred = frcnn([input_image])[0]

if device.type == "cuda":
    torch.cuda.synchronize()

frcnn_time = time.perf_counter() - start

frcnn_output, frcnn_count = draw_detections(
    image,
    frcnn_pred["boxes"].cpu(),
    frcnn_pred["labels"].cpu(),
    frcnn_pred["scores"].cpu(),
    frcnn_categories
)

show_comparison(image, frcnn_output, "Faster R-CNN Output")
print("Detections:", frcnn_count)
print("Inference time:", round(frcnn_time, 3), "seconds")


# 6. Load RetinaNet
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights
)

retina_weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT

retinanet = retinanet_resnet50_fpn_v2(
    weights=retina_weights
).to(device).eval()

retina_categories = retina_weights.meta["categories"]
retina_preprocess = retina_weights.transforms()

print("RetinaNet loaded.")


# 7. Detect objects using RetinaNet
input_image = retina_preprocess(image).to(device)

if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

with torch.no_grad():
    retina_pred = retinanet([input_image])[0]

if device.type == "cuda":
    torch.cuda.synchronize()

retina_time = time.perf_counter() - start

retina_output, retina_count = draw_detections(
    image,
    retina_pred["boxes"].cpu(),
    retina_pred["labels"].cpu(),
    retina_pred["scores"].cpu(),
    retina_categories
)

show_comparison(image, retina_output, "RetinaNet Output")
print("Detections:", retina_count)
print("Inference time:", round(retina_time, 3), "seconds")


# 8. Load YOLO
from ultralytics import YOLO

try:
    model_name = "yolo26n.pt"
    yolo = YOLO(model_name)
except Exception:
    model_name = "yolo11n.pt"
    yolo = YOLO(model_name)

print("Loaded:", model_name)


# 9. Detect objects using YOLO
start = time.perf_counter()

results = yolo.predict(
    source=np.array(image),
    conf=0.40,
    verbose=False
)

yolo_time = time.perf_counter() - start

result = results[0]

# YOLO plot output is BGR; convert to RGB
yolo_output = result.plot()[:, :, ::-1]

show_comparison(image, yolo_output, "YOLO Output")

yolo_count = len(result.boxes)

print("Detections:", yolo_count)
print("Inference time:", round(yolo_time, 3), "seconds")


# 10. Compare all three models
print("\nMODEL COMPARISON")
print("-" * 45)
print(f"Faster R-CNN: {frcnn_count} detections, "
      f"{frcnn_time:.3f} seconds")
print(f"RetinaNet:    {retina_count} detections, "
      f"{retina_time:.3f} seconds")
print(f"YOLO:         {yolo_count} detections, "
      f"{yolo_time:.3f} seconds")
