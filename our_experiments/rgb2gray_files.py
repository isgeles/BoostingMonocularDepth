import os
import numpy as np
import cv2
from PIL import Image


# convert all png rgb files in folder to grayscale
def rgs2gray_files(rgb_dir, img_dir):
    for file in os.listdir(rgb_dir):
        if file.endswith(".png"):
            print("File:", file)
            img = cv2.imread(os.path.join(rgb_dir, file), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img = Image.fromarray(img)
            img = img.convert("L")

            # create output folder
            os.makedirs(img_dir, exist_ok=True)
            img.save(os.path.join(img_dir, file[:-3] + 'png'))


if __name__ == "__main__":
    input_dir = "./IBIMS1/SGR/outputs_original"
    output_dir = "./IBIMS1/SGR/outputs_original_"

    rgs2gray_files(input_dir, output_dir)