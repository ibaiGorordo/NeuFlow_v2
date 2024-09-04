import os
import torch

from utils import load_model, ModelType

model_type = ModelType.MIXED
device = torch.device('cuda')
image_width = 640
image_height = 480
iters_s16=1
iters_s8=8
half = False
model = load_model(model_type, device, (image_height, image_width), iters_s16, iters_s8, half)

dtype = torch.half if half else torch.float
img = torch.randn(1, 3, image_height, image_width, device=device, dtype=dtype)

with torch.no_grad():

    # Test the model
    flow = model(img, img)[0][0]

    model_name =  f"{model_type.value}.onnx"
    torch.onnx.export(model,
                      (img, img),
                      model_name,
                      verbose=False,
                      opset_version=16,
                      input_names=['input1', 'input2'],
                      output_names=['output'])

os.system(f"onnxsim {model_name} {model_name}")