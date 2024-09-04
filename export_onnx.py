import torch
from glob import glob
import os
import numpy as np
import cv2
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from data_utils import flow_viz



def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def load_model(model_path: str, device: torch.device) -> NeuFlow:

    model = NeuFlow().to(device)
    checkpoint = torch.load('neuflow_mixed.pth', map_location='cuda')
    model.load_state_dict(checkpoint['model'], strict=True)
    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
            delattr(m, "norm1")  # remove batchnorm
            delattr(m, "norm2")  # remove batchnorm
            m.forward = m.forward_fuse  # update forward
    model.eval()


    return model

image_width = 640
image_height = 480
model_path = 'neuflow_mixed.pth'
device = torch.device('cuda')
model = load_model('neuflow_mixed.pth', device)
model.init_bhwd(1, image_height, image_width, 'cuda', amp=False)
img = torch.randn(1, 3, image_height, image_width, device=device, dtype=torch.float32)

with torch.no_grad():
    flow = model(img, img)[0][0]
    print(flow.shape)

    torch.onnx.export(model,
                      (img, img),
                      'neuflow.onnx',
                      verbose=True,
                      opset_version=16,
                      dynamic_axes={'input1': {2: 'height', 3: 'width'},
                                    'input2': {2: 'height', 3: 'width'},
                                    'output': {2: 'height', 3: 'width'}},
                      input_names=['input1', 'input2'],
                      output_names=['output'])