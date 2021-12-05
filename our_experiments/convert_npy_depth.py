import numpy as np
import os
import argparse
from PIL import Image


# convert numpy binaries to pngs
def npy_to_png(npy_dir, img_dir):
    """
    Converting many npy (depth) files in a folder to png (depth) images which are grayscale.
    Args:
        npy_dir: directory path of npy input files
        img_dir: directory path to store the output files in png format
    """
    for file in os.listdir(npy_dir):
        if file.endswith(".npy"):
            curr = np.load(os.path.join(npy_dir, file))
            curr = curr.squeeze()

            img = Image.fromarray(curr)
            img = img.convert("L")
            # create output folder
            os.makedirs(img_dir, exist_ok=True)

            #file = file.replace("_depth", '')  # only for DIODE dataset to remove 'depth' from the name
            img.save(os.path.join(img_dir, file[:-3] + 'png'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert npy files in folder to png image files.')
    parser.add_argument('--input_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    args = parser.parse_args()

    npy_to_png(args.input_dir, args.output_dir)