from fastapi import FastAPI, File, UploadFile

import dlib
import cv2
import numpy as np

app = FastAPI()

@app.post("/cam")
def read_root():
    return {"Hello": "World"}