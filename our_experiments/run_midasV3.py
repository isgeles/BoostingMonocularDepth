
from torchvision.transforms import Compose

# OUR
import utils

# MIDAS
import midas.utils
from midas.models.transforms import Resize, NormalizeImage, PrepareForNet

import os
import torch
import cv2
import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)

# select device
#device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: %s" % device)

# Inference on MiDas v3
def estimatemidasV3(img, msize):

    transform = Compose(
        [
            Resize(
                msize,
                msize,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )

    img_input = transform({"image": img})["image"]

    # Forward pass
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = midasmodelv3.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Normalization
    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        prediction = (prediction - depth_min) / (depth_max - depth_min)
    else:
        prediction = 0

    return prediction



def run(dataset, out_dir, msize=384):
    # Go through all images in input directory
    print("start processing")
    for image_ind, images in enumerate(dataset):

        # if os.path.isfile(os.path.join(result_dir, images.name + '.png')):
        #     print('skipping image', image_ind, ':', images.name)
        #     continue

        print('processing image', image_ind, ':', images.name)

        img = images.rgb_image
        output_resolution = img.shape[1], img.shape[0]

        estimation = estimatemidasV3(img, msize)

        path = os.path.join(out_dir, images.name)

        midas.utils.write_depth(path, cv2.resize(estimation, output_resolution, interpolation=cv2.INTER_CUBIC),
                                bits=2, colored=False)



if __name__ == "__main__":
    # model

    # model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    global midasmodelv3
    midasmodelv3 = torch.hub.load("intel-isl/MiDaS", model_type)
    midasmodelv3.to(device)
    midasmodelv3.eval()

    output_dir = "./outputs/midasV3_diode"
    os.makedirs(output_dir, exist_ok=True)

    data_dir = "../dataset_prepare/DIODE/rgb"
    # Create dataset from input images
    dataset = utils.ImageDataset(data_dir, 'test')

    run(dataset, output_dir)