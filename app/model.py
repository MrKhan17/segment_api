# import torch
# import numpy as np
# from PIL import Image

# # Load YOLOv8 model
# def load_model(model_path: str):
#     model = torch.hub.load("ultralytics/yolov8", "custom", path=model_path)
#     return model

# # Process image and run YOLOv8 model inference
# def process_image(model, image: Image):
#     img_array = np.array(image)
#     results = model(img_array)
#     results.render()  # Renders the image with predictions
#     return results.ims  # Assuming `ims` is the list of rendered images
