from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
import base64
import cv2
import numpy as np

app = FastAPI(title="Face Verification API")

# ----------- Request Model -----------
class FaceRequest(BaseModel):
    image1: str
    image2: str

# ----------- Utils -----------
def base64_to_image(base64_string: str):
    try:
        # Remove data:image/...;base64, if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Decoded image is None")

        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 image")


# ----------- API Endpoint -----------
@app.post("/face/verify")
def verify_faces(request: FaceRequest):
    try:
        img1 = base64_to_image(request.image1)
        img2 = base64_to_image(request.image2)

        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name="ArcFace",
            distance_metric="cosine",
            enforce_detection=True
        )

        similarity = 1 - result["distance"]
        confidence = round(similarity * 100, 2)

        return {
            "confidence": confidence,
            "samePerson": result["verified"]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
