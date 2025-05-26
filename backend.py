
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    return np.array(image)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify "http://127.0.0.1"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_PATH = "1.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI server!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image= read_file_as_image(await file.read())
    MODEL.predict(np.expand_dims(image, axis=0))
    predictions = MODEL.predict(np.expand_dims(image, axis=0))
    index=np.argmax(predictions[0])
    pridicted_class = CLASS_NAMES[index]
    confidence=np.max(predictions[0])
    return {
        "filename": file.filename,
        "class": pridicted_class,
        "confidence": float(confidence)}

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="localhost", log_level="info") 
