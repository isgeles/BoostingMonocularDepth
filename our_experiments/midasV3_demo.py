""" adapted code from the Midas repository demo notebook """

import numpy as np
import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

def write_depth(path, depth, bits=1 , colored=False):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    # write_pfm(path + ".pfm", depth.astype(np.float32))
    if colored == True:
        bits = 1

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1
    # if depth_max>max_val:
    #     print('Warning: Depth being clipped')
    #
    # if depth_max - depth_min > np.finfo("float").eps:
    #     out = depth
    #     out [depth > max_val] = max_val
    # else:
    #     out = 0

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1 or colored:
        out = out.astype("uint8")
        if colored:
            out = cv2.applyColorMap(out,cv2.COLORMAP_INFERNO)
        cv2.imwrite(path+'.png', out)
    elif bits == 2:
        cv2.imwrite(path+'.png', out.astype("uint16"))

    return

# insert path or download from url
filename = "../inputs/sample0.jpg"
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)


model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

write_depth("./out1", output, bits=2, colored=True)

plt.imshow(output, cmap='gray')
plt.show()