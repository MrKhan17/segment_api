from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# Pydantic model to define input format
class ImageBase64Input(BaseModel):  
    image_base64: str

# Load YOLOv8 model
model = YOLO("crop_segment.pt")  # Replace with your model's path


@app.post("/predict/")
async def predict(input: ImageBase64Input):
    try:
        bounding_boxes = []
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
            # Access segmentation masks
            if result.masks is not None:
              
                for i, polygon in enumerate(result.masks.xy):
                    # Extract the x and y coordinates
                    x_coords = polygon[0]
                    y_coords = polygon[1]
                    
                    top_left = (float(min(x_coords)), float(min(y_coords)))
                    top_right = (float(max(x_coords)), float(min(y_coords)))
                    bottom_left = (float(min(x_coords)), float(max(y_coords)))
                    bottom_right = (float(max(x_coords)), float(max(y_coords)))
                
                    # Append the bounding box to the list
                    bounding_boxes.append((
                            top_left,
                            top_right,
                            bottom_left,
                            bottom_right
                                  )
                            )
        return {"segmented_image_coordinates": bounding_boxes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
