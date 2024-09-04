import torch

from utils import load_model, ModelType

model_type = ModelType.MIXED
image_width = 640
image_height = 480
device = torch.device('cuda')
half = False
model = load_model(model_type, device, (image_height, image_width), half)

dtype = torch.half if half else torch.float
img = torch.randn(1, 3, image_height, image_width, device=device, dtype=dtype)

with torch.no_grad():

    # Test the model
    flow = model(img, img)[0][0]

    model_name =  f"{model_type.value}.onnx"
    torch.onnx.export(model,
                      (img, img),
                      model_name,
                      verbose=True,
                      opset_version=16,
                      input_names=['input1', 'input2'],
                      output_names=['output'])