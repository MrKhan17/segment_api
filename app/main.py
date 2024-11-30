from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
# from io import BytesIO
# from PIL import Image
# import torch
# from .model import load_model, process_image
import numpy as np
import cv2
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# Pydantic model to define input format
class ImageBase64Input(BaseModel):
    image_base64: str

# Load YOLOv8 model
# model = load_model("/Users/aidanamoldazaiym/Downloads/crop_segment.pt")
model = YOLO("/Users/aidanamoldazaiym/Downloads/crop_segment.pt")  # Replace with your model's path


# API endpoint to process the image and return the output
@app.post("/predict/")
async def predict(input: ImageBase64Input):
    try:
        # Convert base64 image to PIL Image
        image_data = base64.b64decode(input.image_base64)
        # Convert bytes to a NumPy array
        nparr = np.frombuffer(image_data, np.uint8)

        # Decode the array into an image (OpenCV format)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # image = Image.open(BytesIO(image_data))
        results = model(image, task='segment',save=True) 
        # Iterate through results for each image
        for idx, result in enumerate(results):
            print(f"Image {idx + 1}:")
            
            # Access segmentation masks
            if result.masks is not None:
                for i, polygon in enumerate(result.masks.xy):
                    print(f"Mask {i} coordinates for Image {idx + 1}: {polygon}")
        # Process image with YOLOv8 model
        # result = process_image(model, image)

        # Convert output to base64 for return
        # output_image = result[0]  # Assuming the result has the image with segmentation overlay
        # buffered = BytesIO()
        # output_image.save(buffered, format="PNG")
        # output_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"segmented_image": 'akefjnkj'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
