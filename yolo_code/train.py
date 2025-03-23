from ultralytics import YOLO

# Load a model
# model = YOLO("yolo8n.yaml")  # build a new model from YAML
# model = YOLO("yolo8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo8n.yaml").load("yolo8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8.yaml", epochs=100, batch=16)