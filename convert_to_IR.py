# import platform

# %pip install -q "openvino>=2023.3.0" "nncf>=2.8.1" "opencv-python" "seaborn" "pandas" "scikit-learn" "torch" "torchvision"  --extra-index-url https://download.pytorch.org/whl/cpu

# if platform.system() != "Windows":
#     %pip install -q "matplotlib>=3.4"
# else:
#     %pip install -q "matplotlib>=3.4,<3.7"







from models.experimental import attempt_load
import torch, os
import openvino as ov
from models.yolo import Detect, DualDDetect
from utils.general import yaml_save, yaml_load


MODEL_DIR = "/home/ai-ctrl/matriceAI/yolov9Training_matrice-20240413T092205Z-001/yolov9Training_matrice/yolov9/runs/train/exp4/weights/"
weights = MODEL_DIR+"gelan-c.pt"
ov_model_path = MODEL_DIR + weights.replace(".pt", "_openvino_model") + "/"+ weights.replace(".pt", ".xml")

if not os.path.exists(ov_model_path):
    model = attempt_load(weights, device="cpu", inplace=True, fuse=True)
    metadata = {'stride': int(max(model.stride)), 'names': model.names}

    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, (Detect, DualDDetect)):
            m.inplace = False
            m.dynamic = True
            m.export = True


    example_input = torch.zeros((1, 3, 640, 640))
    print("passing input to model")
    # model(example_input)
    print("getting output from model")

    ov_model = ov.convert_model(model, example_input=example_input)

    # specify input and output names for compatibility with yolov9 repo interface
    ov_model.outputs[0].get_tensor().set_names({"output0"})
    ov_model.inputs[0].get_tensor().set_names({"images"})
    ov.save_model(ov_model, ov_model_path)
    # save metadata
    yaml_save(ov_model_path.parent / weights.replace(".pt", ".yaml"), metadata)
else:
    metadata = yaml_load(ov_model_path.parent + weights.replace(".pt", ".yaml"))







# import numpy as np
# import torch
# from PIL import Image
# from utils.augmentations import letterbox

# image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/7b6af406-4ccb-4ded-a13d-62b7c0e42e96"
# download_file(image_url, directory=DATA_DIR, filename="test_image.jpg", show_progress=True)

# def preprocess_image(img0: np.ndarray):
#     """
#     Preprocess image according to YOLOv9 input requirements.
#     Takes image in np.array format, resizes it to specific size using letterbox resize, converts color space from BGR (default in OpenCV) to RGB and changes data layout from HWC to CHW.

#     Parameters:
#       img0 (np.ndarray): image for preprocessing
#     Returns:
#       img (np.ndarray): image after preprocessing
#       img0 (np.ndarray): original image
#     """
#     # resize
#     img = letterbox(img0, auto=False)[0]

#     # Convert
#     img = img.transpose(2, 0, 1)
#     img = np.ascontiguousarray(img)
#     return img, img0


# def prepare_input_tensor(image: np.ndarray):
#     """
#     Converts preprocessed image to tensor format according to YOLOv9 input requirements.
#     Takes image in np.array format with unit8 data in [0, 255] range and converts it to torch.Tensor object with float data in [0, 1] range

#     Parameters:
#       image (np.ndarray): image for conversion to tensor
#     Returns:
#       input_tensor (torch.Tensor): float tensor ready to use for YOLOv9 inference
#     """
#     input_tensor = image.astype(np.float32)  # uint8 to fp16/32
#     input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

#     if input_tensor.ndim == 3:
#         input_tensor = np.expand_dims(input_tensor, 0)
#     return input_tensor

# NAMES = metadata["names"]





# from utils.plots import Annotator, colors

# from typing import List, Tuple
# from utils.general import scale_boxes, non_max_suppression


# def detect(model: ov.Model, image_path: Path, conf_thres: float = 0.25, iou_thres: float = 0.45, classes: List[int] = None, agnostic_nms: bool = False):
#     """
#     OpenVINO YOLOv9 model inference function. Reads image, preprocess it, runs model inference and postprocess results using NMS.
#     Parameters:
#         model (Model): OpenVINO compiled model.
#         image_path (Path): input image path.
#         conf_thres (float, *optional*, 0.25): minimal accepted confidence for object filtering
#         iou_thres (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
#         classes (List[int], *optional*, None): labels for prediction filtering, if not provided all predicted labels will be used
#         agnostic_nms (bool, *optional*, False): apply class agnostic NMS approach or not
#     Returns:
#        pred (List): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
#        orig_img (np.ndarray): image before preprocessing, can be used for results visualization
#        inpjut_shape (Tuple[int]): shape of model input tensor, can be used for output rescaling
#     """
#     if isinstance(image_path, np.ndarray):
#         img = image_path
#     else:
#         img = np.array(Image.open(image_path))
#     preprocessed_img, orig_img = preprocess_image(img)
#     input_tensor = prepare_input_tensor(preprocessed_img)
#     predictions = torch.from_numpy(model(input_tensor)[0])
#     pred = non_max_suppression(predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
#     return pred, orig_img, input_tensor.shape


# def draw_boxes(predictions: np.ndarray, input_shape: Tuple[int], image: np.ndarray, names: List[str]):
#     """
#     Utility function for drawing predicted bounding boxes on image
#     Parameters:
#         predictions (np.ndarray): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
#         image (np.ndarray): image for boxes visualization
#         names (List[str]): list of names for each class in dataset
#         colors (Dict[str, int]): mapping between class name and drawing color
#     Returns:
#         image (np.ndarray): box visualization result
#     """
#     if not len(predictions):
#         return image

#     annotator = Annotator(image, line_width=1, example=str(names))
#     # Rescale boxes from input size to original image size
#     predictions[:, :4] = scale_boxes(input_shape[2:], predictions[:, :4], image.shape).round()

#     # Write results
#     for *xyxy, conf, cls in reversed(predictions):
#         label = f'{names[int(cls)]} {conf:.2f}'
#         annotator.box_label(xyxy, label, color=colors(int(cls), True))
#     return image




# core = ov.Core()
# # read converted model
# ov_model = core.read_model(ov_model_path)


# import ipywidgets as widgets

# device = widgets.Dropdown(
#     options=core.available_devices + ["AUTO"],
#     value='AUTO',
#     description='Device:',
#     disabled=False,
# )

# device