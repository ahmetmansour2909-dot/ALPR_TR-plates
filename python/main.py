from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
from ocr_model import OCRModel
from PIL import Image
import io

app = FastAPI()

ocr = OCRModel("ocr_resnet_ctc.pth")

@app.post("/ocr")
async def read_plate(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = ocr.predict(image)
    return {"plate": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
