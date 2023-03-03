import requests
from io import BytesIO
from PIL import Image
import json
import numpy as np
from depthai_sdk import DepthAI

# Specify your text here
text = input("Enter your text here: ")

# Generate an image from text using DALL-E 2
response = requests.post(
    "https://api.openai.com/v1/images/generations",
    headers={
        "Content-Type": "application/json",
        "Authorization": "YOUR_API_KEY",

    },
    data=json.dumps({
        "model": "image-alpha-001",
        "prompt": text,
        "num_images": 1,
        "size": "512x512",
        "response_format": "url"
    })
)

# Retrieve the URL of the generated image
image_url = response.json()["data"][0]["url"]

# Download the image from the URL
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Convert the image to a 3D image using DepthAI
depthai = DepthAI()
depthai.load_model("model.blob") # Replace with the path to your model file
depth_image = depthai.generate_depth_map(np.array(image))

# Save the depth image
depth_image.save("depth_image.png")
