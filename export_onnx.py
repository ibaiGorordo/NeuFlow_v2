import os
import torch
import argparse
from utils import load_model, ModelType

def get_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--model', type=str, default='mixed', choices=['mixed', 'sintel', 'things'], help='Model type')
    parser.add_argument('--half', action='store_true', help='Use half precision')
    parser.add_argument('--image_width', type=int, default=768, help='Image width')
    parser.add_argument('--image_height', type=int, default=432, help='Image height')
    parser.add_argument('--iters_s16', type=int, default=1, help='Number of iterations for s16')
    parser.add_argument('--iters_s8', type=int, default=8, help='Number of iterations for s8')

    return parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cpu')

    args = get_args()
    model_type = ModelType[args.model.upper()]
    image_width = args.image_width
    image_height = args.image_height
    iters_s16=args.iters_s16
    iters_s8=args.iters_s8
    half = args.half
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