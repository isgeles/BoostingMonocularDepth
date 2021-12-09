# Our Additions and Experiments with the Baseline Repository

## Adjusted the code to run on CPU only

Some lines in various files needed to be adjusted to use the device 'cpu' instead of a default GPU option for CUDA.
The changes in a few files were uncommenting the default 'cuda' (GPU) option and adding the 'cpu' device option such as shown here
```py
# device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
or 
```py 
# input = rgb.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input = rgb.to(device)
```


## Testing implemented methods for different models with IBIMS-1 dataset
Download the IBIMS-1 and place all rgb images in a folder. 
- SGR: place downloaded weights in folder, compile module and run the code as described by the authors demo code
- MiDaS v2: download weights and place in midas folder, run as described by the authors
- LeReS: download weights and place in root repository directory, run as described by the authors

## Adding another depth model (MiDaS v3) to the code baseline
Demo code for MiDaS v3 was adjusted from the MiDaS repository, such that it runs a whole folder of images in `run_midasV3.py`.
It automatically downloads the right weights from torchhub and the right transformation for the original estimation.

## Testing SGR, MiDaS v2 and MiDaS v3 with the DIODE dataset

Similar to before, just changing the input directory to the rgb image directory of DIODE.

### Pre-processing the ground truth data of DIODE

In the `convert_npy_depth.py` file we convert the DIODE npy depth data to png such that the evaluation method does not need to be changed.


## Computing R20 for any given image

## Using Demo Code of SGR, MiDaS and LeReS (taken from each models repository and adjusted) to run model on it own

## Evaluation script modifications

In the case of LeReS, depth is estimated instead of disparity, therefore we swap the two variables when LeReS is evaluated by adding:

```
if dataset_disp_esttype
    temp = estimate_disp;
    estimate_disp = estimate_depth;
    estimate_depth = temp;
end
```

Moreover, a simple check was added to skip pictures not present in the estimation path.



