try:
    import unzip_requirements
except ImportError:
    pass

import base64
import boto3
import os
import io
import json
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from io import BytesIO
from PIL import Image
from requests_toolbelt.multipart import decoder

S3_BUCKET = os.environ["S3_BUCKET"] if "S3_BUCKET" in os.environ else "sgm-models"
MODEL_PATH = (
    os.environ["MODEL_PATH"] if "MODEL_PATH" in os.environ else "./mobilenet_v2.pt"
)

print(f"Donwloading model: {MODEL_PATH}")

# Loading the S3 client when Lambda execution context is created
s3 = boto3.client("s3")


def load_model_from_s3():
    try:
        if os.path.isfile(MODEL_PATH) != True:
            obj = s3.get_object(Bucket=S3_BUCKET, key=MODEL_PATH)
            print("Creating Byte stream")
            byte_stream = io.BytesIO(obj["Body"].read())
            print("Loading model.")
            model = torch.jit.load(byte_stream)
            print("Model loaded")
            return model
        else:
            print("Model loading failed.")
    except Exception as e:
        print(repr(e))
        raise (e)


model = load_model_from_s3()


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise (e)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()


def classify_image(event, context):
    try:
        content_type_header = event["headers"]["content-type"]
        print(f"Image content: {event['body']}")

        body = base64.b64decode(event["body"])
        print("Body Loaded.")

        img = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = get_prediction(image_bytes=img.content)
        print(f"Image classification code: {prediction}")

        file_name = (
            img.headers[b"Content-Disposition"].decode().split(";")[2].split("=")[1]
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.cump(
                {"file": file_name.replace('"', ""), "predicted": prediction}
            ),
        }

    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.dumps({"error": repr(e)}),
        }
