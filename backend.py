
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

MODEL_PATH = "1.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).resize((256, 256))
    return np.array(image)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)
    predictions = MODEL.predict(image_batch)
    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]
    confidence = float(np.max(predictions[0]))
    return {
        "filename": file.filename,
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
