import time
import torch
import cv2
from utils import load_model, process_image, flow_to_image

image_width = 768
image_height = 432

model_path = 'models/neuflow_mix.pth'
device = torch.device('cuda')
model = load_model(model_path, device, (image_height, image_width))


for image_path_0, image_path_1 in zip(image_path_list[:-1], image_path_list[1:]):

    print(image_path_0)

    image_0 = process_image(image_path_0)
    image_1 = process_image(image_path_1)

    with torch.no_grad():
        start = time.perf_counter()
        flow = model(image_0, image_1)[-1][0]
        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")

        flow = flow.permute(1,2,0).cpu().numpy()
        
        flow = flow_to_image(flow)

        image_0 = cv2.resize(cv2.imread(image_path_0), (image_width, image_height))
