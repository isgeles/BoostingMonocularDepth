""" Running simply LeRes with a single image or a folder of images. """

from torchvision.transforms import transforms

# OUR
import utils

# MIDAS
import midas.utils

#AdelaiDepth
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import strip_prefix_if_present

import os
import torch
import cv2
import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: %s" % device)


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img.astype(np.float32))
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


# Inference on LeRes
def estimateleres(img, msize):
    # LeReS forward pass script adapted from https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS

    rgb_c = img[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (msize, msize))
    img_torch = scale_torch(A_resize)[None, :, :, :]

    try:
        # Forward pass
        with torch.no_grad():
            prediction = leresmodel.inference(img_torch)
    except RuntimeError as err:
        printf('CUDA ran out of memory, retrying...')
        gc.collect() # Python cleanup
        with torch.no_grad():
            prediction = leresmodel.inference(img_torch)
    catch Exception as err:
        printf('other error occured')
        raise

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    return prediction


def run(dataset, out_dir, msize=448):
    # Go through all images in input directory
    print("start processing")
    for image_ind, images in enumerate(dataset):

        # if os.path.isfile(os.path.join(result_dir, images.name + '.png')):
        #     print('skipping image', image_ind, ':', images.name)
        #     continue

        print('processing image', image_ind, ':', images.name)

        img = images.rgb_image
        output_resolution = img.shape[1], img.shape[0]

        estimation = estimateleres(img, msize)

        path = os.path.join(out_dir, images.name)

        midas.utils.write_depth(path, cv2.resize(estimation, output_resolution, interpolation=cv2.INTER_CUBIC),
                                bits=2, colored=False)



if __name__ == "__main__":


    # leres model
    global leresmodel
    leres_model_path = "../res101.pth"
    checkpoint = torch.load(leres_model_path, map_location=torch.device(device))
    leresmodel = RelDepthModel(backbone='resnext101')
    leresmodel.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."),
                               strict=True)
    del checkpoint
    torch.cuda.empty_cache()
    leresmodel.to(device)
    leresmodel.eval()

    #### run single image
    # img_path = "./inputs/IMG_20211014_133707.jpg"
    # msize = 448
    # colorize_results = True
    # output_path = "./outputs/leres_out"
    # img = utils.read_image(img_path)
    # output_size = img.shape[1], img.shape[0]
    #
    # estimation = estimateleres(img, msize)
    #
    # midas.utils.write_depth(output_path, cv2.resize(estimation, output_size,
    #                         interpolation=cv2.INTER_CUBIC), bits=2, colored=colorize_results)
    ####

    # running whole folder with images

    output_dir = "./outputs/leres_ibims1"
    os.makedirs(output_dir, exist_ok=True)

    data_dir = "../dataset_prepare/ibims1/rgb"
    # Create dataset from input images
    dataset = utils.ImageDataset(data_dir, 'test')

    run(dataset, output_dir)



