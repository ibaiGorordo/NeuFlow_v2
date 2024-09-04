import time
import torch
import cv2
from imread_from_url import imread_from_url

from utils import load_model, process_image, flow_to_image, ModelType

image_width = 768
image_height = 432

model_type = ModelType.MIXED
device = torch.device('cuda')
model = load_model(model_type, device, (image_height, image_width))

img1 = imread_from_url("https://github.com/princeton-vl/RAFT/blob/master/demo-frames/frame_0016.png?raw=true")
img2 = imread_from_url("https://github.com/princeton-vl/RAFT/blob/master/demo-frames/frame_0025.png?raw=true")

input1 = process_image(img1)
input2 = process_image(img2)

with torch.inference_mode():
    start = time.perf_counter()
    flow = model(input1, input2)[-1][0]
    print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")

    flow = flow.permute(1,2,0).cpu().numpy()
    flow = flow_to_image(flow)
    flow = cv2.cvtColor(flow, cv2.COLOR_RGB2BGR)
    flow = cv2.resize(flow, (img1.shape[1], img1.shape[0]))

    combined = cv2.vconcat([img1, img2, flow])
    cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)
    cv2.imshow("Optical Flow", combined)
    cv2.waitKey(0)