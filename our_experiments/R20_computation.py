import numpy as np
import cv2
from skimage import measure


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255.0
    #if img.ndim == 2:
    #    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img.astype(np.float32)


def rgb2gray(rgb):
    # Converts rgb to gray
    # Y' = 0.2989 R + 0.5870 G + 0.1140 B
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def resizewithpool(img, size):
    i_size = img.shape[0]
    n = int(np.floor(i_size/size))

    out = measure.block_reduce(img, (n, n), np.max)
    return out


def calculateprocessingres(img, basesize, confidence=0.1, scale_threshold=3, whole_size_threshold=3000):
    # Returns the R_x resolution described in section 5 of the main paper.

    # Parameters:
    #    img :input rgb image
    #    basesize : size the dilation kernel which is equal to receptive field of the network.
    #    confidence: value of x in R_x; allowed percentage of pixels that are not getting any contextual cue.
    #    scale_threshold: maximum allowed upscaling on the input image ; it has been set to 3.
    #    whole_size_threshold: maximum allowed resolution. (R_max from section 6 of the main paper)

    # Returns:
    #    outputsize_scale*speed_scale :The computed R_x resolution
    #    patch_scale: K parameter from section 6 of the paper

    # speed scale parameter is to process every image in a smaller size to accelerate the R_x resolution search
    speed_scale = 32
    image_dim = int(min(img.shape[0:2]))

    gray = rgb2gray(img)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad = cv2.resize(grad, (image_dim, image_dim), cv2.INTER_AREA)

    # thresholding the gradient map to generate the edge-map as a proxy of the contextual cues
    m = grad.min()
    M = grad.max()
    middle = m + (0.4 * (M - m))
    grad[grad < middle] = 0
    grad[grad >= middle] = 1

    # dilation kernel with size of the receptive field
    kernel = np.ones((int(basesize/speed_scale), int(basesize/speed_scale)), np.float)
    # dilation kernel with size of the a quarter of receptive field used to compute k
    # as described in section 6 of main paper
    kernel2 = np.ones((int(basesize / (4*speed_scale)), int(basesize / (4*speed_scale))), np.float)

    # Output resolution limit set by the whole_size_threshold and scale_threshold.
    threshold = min(whole_size_threshold, scale_threshold * max(img.shape[:2]))

    outputsize_scale = basesize / speed_scale
    for p_size in range(int(basesize/speed_scale), int(threshold/speed_scale), int(basesize / (2*speed_scale))):
        grad_resized = resizewithpool(grad, p_size)
        grad_resized = cv2.resize(grad_resized, (p_size, p_size), cv2.INTER_NEAREST)
        grad_resized[grad_resized >= 0.5] = 1
        grad_resized[grad_resized < 0.5] = 0

        dilated = cv2.dilate(grad_resized, kernel, iterations=1)
        meanvalue = (1-dilated).mean()
        if meanvalue > confidence:
            break
        else:
            outputsize_scale = p_size

    grad_region = cv2.dilate(grad_resized, kernel2, iterations=1)
    patch_scale = grad_region.mean()

    return int(outputsize_scale*speed_scale), patch_scale


############################################################################
#### Set the image name that you want to optimize for
image_path = '../inputs/sample2.jpg'
############################################################################

img = read_image(image_path)

# Hyperparameters
whole_size_threshold = 3000  # R_max from the paper
r_threshold_value = 0.2 # Value x of R_x defined in the section 5 of the main paper.
scale_threshold = 3  # Allows up-scaling with a scale up to 3

##########################################################################
##### Fill in the receptive field size of your network here ##############
# MiDas: 384
# SGR: 448
# LeRes: 448
net_receptive_field_size = 384
##########################################################################

# Find the best input resolution R-x.
# The resolution search described in section 5-double estimation of the main paper and section B of the
# supplementary material.
whole_image_optimal_size, _ = calculateprocessingres(img, net_receptive_field_size,r_threshold_value,
                                                     scale_threshold, whole_size_threshold)

print("Optimal resolution R20 is: ", whole_image_optimal_size)