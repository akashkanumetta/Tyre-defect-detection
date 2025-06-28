import io
import base64
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tensorflow as tf 
import uvicorn

model = tf.keras.models.load_model(r'C:\Users\msdak\Desktop\vscode\cnn_model1.h5')

app = FastAPI()

class Image(BaseModel):
    image: str

def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def draw_label_and_display(image: np.ndarray, pred, bbox_size=(100, 50), bbox_pos=(50, 50)):
    label = 'Defect' if pred[0][0] >= 0.5 else 'OK'

    x, y, w, h = bbox_pos[0], bbox_pos[1], bbox_size[0], bbox_size[1]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0) if label == 'OK' else (0, 0, 255), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if label == 'OK' else (0, 0, 255), 2)

    return image

@app.post("/predict")
async def predict(image: Image):
    model = tf.keras.models.load_model(r'C:\Users\msdak\Desktop\vscode\cnn_model1.h5')
    model.eval()

    image_data = base64.b64decode(image.image.split(',')[1])
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    preprocessed_image = preprocess_image(image)
    pred = model(torch.tensor(preprocessed_image, dtype=torch.float32)).detach().numpy()

    image_with_bbox = draw_label_and_display(image, pred)

    _, buffer = cv2.imencode('.jpg', image_with_bbox)
    bbox_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "label": "Defect" if pred[0][0] >= 0.5 else "GOOD",
        "bbox_image": bbox_base64
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)