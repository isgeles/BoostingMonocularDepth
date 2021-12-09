# OUR
import utils

# MIDAS
import midas.utils

# PIX2PIX : MERGE NET
from pix2pix.options.test_options import TestOptions
from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

import os
import torch
import cv2
import numpy as np
import warnings

warnings.simplefilter('ignore', np.RankWarning)

# select device
# device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: %s" % device)


# Generate a double-input depth estimation
def merge2pics(img1, img2, pix2pixsize):

    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(img1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC).squeeze()

    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(img2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC).squeeze()

    # Inference on the merge model
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped + 1) / 2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
            torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped



if __name__ == "__main__":

    PIX2PIXSIZE = 1024
    img2_path = "./inputs/merge2/IMG_midasV2.png"
    img1_path = "./inputs/merge2/IMG_midasV3.png"
    output_path = "./inputs/merge2"
    colorize_results = True

    os.makedirs(output_path, exist_ok=True)

    img1 = cv2.imread(img1_path)
    img1 = utils.rgb2gray(img1) / 255.0

    img2 = cv2.imread(img2_path)
    img2 = utils.rgb2gray(img2) / 255.0

    outputsize = img1.shape[1], img1.shape[0]

    img1 = cv2.resize(img1, (PIX2PIXSIZE, PIX2PIXSIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img2 = cv2.resize(img2, (PIX2PIXSIZE, PIX2PIXSIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    # Load merge network
    opt = TestOptions().parse()
    global pix2pixmodel
    pix2pixmodel = Pix2Pix4DepthModel(opt)
    pix2pixmodel.save_dir = '../pix2pix/checkpoints/mergemodel'
    pix2pixmodel.load_networks('latest')
    pix2pixmodel.eval()

    merged_img = merge2pics(img1, img2, PIX2PIXSIZE)

    path = os.path.join(output_path, "img1_img2_merge_")

    midas.utils.write_depth(path, cv2.resize(merged_img, outputsize,
                            interpolation=cv2.INTER_CUBIC), bits=2, colored=colorize_results)

    print("Finished merging")