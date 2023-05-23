# GroundingDINO Service
## Why do this
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) is an open-set object detector, which can detect objects that are not in the training set, it can be used to automatic annotation or other applications. However, it is not a service, and it is not easy to use. This project is to provide a service for GroundingDINO, so that it can be used easily with http requests.
## Basic Background Knowledge
We use [TorchServe](https://github.com/pytorch/serve) to serve GroundingDINO model. TorchServe is a flexible and easy to use tool for serving PyTorch models. TorchServe also provides a flexible serialization format and example libraries for TorchServe to serve PyTorch models. TorchServe is a great tool for serving PyTorch models in production and it is easy to use. We can also easily deploy our other model with TorchServe instead of write a new service for each model. For more information about TorchServe, please refer to [TorchServe](https://github.com/pytorch/serve).
## How to use
### 1. clone our repo and cd into it

### 2. Download GroundingDINO and Bert model
Download GroundingDINO model:  
```bash
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
suppose the model is in the weights folder.  
Download Bert model from huggingface:  
https://huggingface.co/bert-base-uncased/tree/main , in this project, we need:
```
config.json
pytorch_model.bin
tokenizer_config.json
tokenizer.json
vocab.txt
```
suppose these files are in the bert-base-uncased folder.
### 3. Build docker image or install dependencies locally
```bash
docker build -t torchserve:groundingdino .
```
or you can use the image I have built:
```bash
docker pull haoliuhust/torchserve:groundingdino
```
### 4. convert model to torchserve format
```bash
docker run --rm -it -v $(pwd):/data -w /data torchserve:groundingdino bash -c "torch-model-archiver --model-name groundingdino --version 1.0 --serialized-file weights/groundingdino_swint_ogc.pth --handler grounding_dino_handler.py --extra-files GroundingDINO_SwinT_OGC.py,bert-base-uncased/*"
```
after it done, you will get a file named groundingdino.mar in the current folder.   
make a folder named model_store, and put the model in it.  
### 5. start torchserve
modify torchserve configurations in config.properties, for more information, please refer to https://github.com/pytorch/serve/blob/master/docs/configuration.md , then start torchserve(change the port as you set in config.properties)
```bash
docker run -d --name groundingdino -v $(pwd)/model_store:/model_store -p 8080:8080 -p 8081:8081 -p 8082:8082 torchserve:groundingdino bash -c "torchserve --start --foreground --model-store /model_store --models groundingdino=groundingdino.mar"
```
### 6. test and use
```python
import requests
import base64
import time
# URL for the web service
url = "http://ip:8080/predictions/groundingdino"
headers = {"Content-Type": "application/json"}

# Input data
with open("test.jpg", "rb") as f:
    image = f.read()

data = {
        "image": base64.b64encode(image).decode("utf-8"), # base64 encoded image or BytesIO
        "caption": "steel pipe", # text prompt, split by "." for multiple phrases
        "box_threshold": 0.25, # threshold for object detection
        "caption_threshold": 0.25 # threshold for text similarity
        }

# Make the request and display the response

resp = requests.post(url=url, headers=headers, json=data)
outputs = resp.json()
'''
the outputs will be like:
    {
        "boxes": [[0.0, 0.0, 1.0, 1.0]], # list of bounding boxes in xyxy format
        "scores": [0.9999998807907104], # list of object detection scores
        "phrases": ["steel pipe"] # list of text phrases
    }

'''
```
## License
The code is licensed under the Apache 2.0 license.

## Reference
[1] [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)  
[2] [TorchServe](https://github.com/pytorch/serve)  
[3] [segment-anything-services](https://github.com/developmentseed/segment-anything-services)
