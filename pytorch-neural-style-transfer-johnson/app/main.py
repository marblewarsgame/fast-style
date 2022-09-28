import string
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn

# Core Imports
import time
import argparse
import random
import shutil
from typing import List
import io

# External Dependency Imports
from imageio import imwrite
import torch
import numpy as np


##############
from stylization_script import argParser, stylize_static_image

app = FastAPI()


# @app.post("/")
# def hello():
#     return "Hello World"


@app.post("/upload", )
async def upload(content: UploadFile = File(...), model: str = "starry_v3.pth"):
   # Check file type
    if content.content_type != "image/png" and content.content_type != "image/jpeg":
        return {"message": "Your Input File has to be in .png or .jpg format. Please try again."}

    if type(model) != str:
        return {{"message": "Model must be in String format. Please try again."}}

    # make unique name using users file name plus random number
    content_path = str(random.randint(1, 10000000)) + content.filename

    # writes content file to local system with the random contetn name
    with open(content_path, "wb") as buffer:
        shutil.copyfileobj(content.file, buffer)

    # set output path to a large random int so its unique
    rand_output_path = str(random.randint(1, 10000000))+".jpg"

    # use styletransfer function
    output_path = argParser(
        content_images_path=content_path, model_name=model, output_images_path=rand_output_path)

    # save output image where ever it needs to be saved

    # Load output image as a temporary file

    # Delete locally hosted output image

    return FileResponse(output_path)

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host="127.0.0.1")
    #uvicorn.run(app, port=8000, host="0.0.0.0")
