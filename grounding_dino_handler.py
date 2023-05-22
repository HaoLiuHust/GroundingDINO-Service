import logging
import os,sys
import uuid
import base64
from PIL import Image
import io
import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import preprocess_caption
from torchvision.ops import box_convert

logger = logging.getLogger(__name__)
console_logger = logging.StreamHandler(sys.stdout)
console_logger.setLevel(logging.DEBUG)
console_logger.setFormatter(logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s"))
logger.addHandler(console_logger)

# file_logger = logging.FileHandler("grounding_dino_handler.log")
# file_logger.setLevel(logging.DEBUG)
# file_logger.setFormatter(logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s"))
# logger.addHandler(file_logger)

class GroundingDinoHandler(object):
    image_preprocessing = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    def __init__(self):
        self.context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        #  load the model
        logger.info("initialize grounding dino handler")
        self.context = context
        self.manifest = context.manifest
        properties = context.system_properties

        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        model_config_path = os.path.join(model_dir, "GroundingDINO_SwinT_OGC.py")
        args = SLConfig.fromfile(model_config_path)
        args.device = self.device
        args.text_encoder_type = model_dir
        self.model = build_model(args)
        checkpoint = torch.load(model_pt_path, map_location="cpu")
        load_res = self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.initialized = True
        logger.info("model loaded successfully")

    def preprocess(self, data):
        #for now, we only support one image per request
        logger.info("preprocess data")

        row = data[0]
        input = row.get("data") or row.get("body")
        if isinstance(input, dict) and "caption" in input and "image" in input \
            and "box_threshold" in input and "caption_threshold" in input:
            image = input["image"]
            request_params = {
                "caption": input["caption"],
                "box_threshold": input["box_threshold"],
                "caption_threshold": input["caption_threshold"]
            }
        else:
            logger.error("No caption or image found in the request")
            assert False, "No caption or image found in the request"
        if isinstance(image, str):
            # if the image is a string of bytesarray.
            image = base64.b64decode(image)
        # If the image is sent as bytesarray
        if isinstance(image, (bytearray, bytes)):
            image = Image.open(io.BytesIO(image))
            request_params["image_size"] = image.size
            image,_ = self.image_preprocessing(image, target=None)
        else:
            # if the image is a list
            image = torch.FloatTensor(image)
            request_params["image_size"] = [image.shape[-1], image.shape[-2]]
        image = image.to(self.device)
        request_params["caption"] = preprocess_caption(request_params["caption"])
        logger.info("preprocess data done")
        return image, request_params
    
    def inference(self, data, *args, **kwargs):
        image, request_params = data
        
        with torch.no_grad():
            outputs = self.model(image[None], captions=[request_params["caption"]])
        return outputs

    def postprocess(self, inference_outputs):
        outputs, request_params = inference_outputs
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)
        box_threshold = request_params["box_threshold"]
        text_threshold = request_params["caption_threshold"]
        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        tokenizer = self.model.tokenizer
        tokenized = tokenizer(request_params["caption"])

        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]
        return boxes, logits.max(dim=1)[0], phrases
    
    def handle(self, data, context):
        self.context = context
        image, request_params = self.preprocess(data)
        outputs = self.inference((image, request_params))
        boxes, scores, phrases = self.postprocess((outputs, request_params))
        #map box to original image
        w, h = request_params["image_size"]
        boxes = boxes * torch.tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        boxes = xyxy.tolist()
        if isinstance(scores, torch.Tensor):
            scores = scores.tolist()
        if isinstance(phrases, torch.Tensor):
            phrases = phrases.tolist()

        logger.info("handle data done")
        return [{"boxes": boxes, "scores": scores, "phrases": phrases}]

if __name__=="__main__":
    import addict
    context = addict.Dict()
    context.system_properties = {
        "gpu_id": 0,
        "model_dir": "/mnt/pai-storage-8/jieshen/code/GroundingDINO/model_store"

    }
    context.manifest = {
        "model": {
            "serializedFile": "groundingdino_swint_ogc.pth"
        }
        }
    handler = GroundingDinoHandler()
    handler.initialize(context)

    with open("/mnt/pai-storage-8/jieshen/work/torchserve/007050.jpg", "rb") as f:
        image = f.read()

    data = [
        {
            "data": {
                "image": image,
                "caption": "steel pipe",
                "box_threshold": 0.25,
                "caption_threshold": 0.25
            }
        }
    ]

    outputs = handler.handle(data, context)
    #draw the boxes
    import cv2
    import numpy as np
    image = cv2.imread("/mnt/pai-storage-8/jieshen/work/torchserve/007050.jpg")
    for box in outputs[0]["boxes"]:
        box = [int(x) for x in box]
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imwrite("test.jpg", image)

    print("test")