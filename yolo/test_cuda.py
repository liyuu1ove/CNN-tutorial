import torch

print(torch.__version__)  # pytorch版本
print(torch.cuda.is_available())  # GPU可用性
print(torch.cuda.device_count())  # GPU个数
print(torch.backends.cudnn.version())  # 查看对应CUDA的版本号
print(torch.version.cuda)  # 查看对应CUDA的版本号

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["asset/zidane.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen