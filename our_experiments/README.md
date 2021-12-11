# Our Additions and Experiments with the Baseline Repository

## Adjusted the code to run on CPU only
Some lines in various files needed to be adjusted to use the device 'cpu' instead of a default GPU option for CUDA.
The changes in a few files were uncommenting the default 'cuda' (GPU) option and adding the 'cpu' device option such as shown here
```py
# device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
or here
```py 
# input = rgb.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input = rgb.to(device)
```

## Adding more command line arguments
We also added the option `--save_lowhigh` which saves the low and high resolution estimates of the base before merging

## Testing implemented methods for different models with IBIMS-1 dataset
Download the IBIMS-1 and place all rgb images in a folder. 
- SGR: place downloaded weights in folder, compile module and run the code as described by the authors demo code
- MiDaS v2: download weights and place in midas folder, run as described by the authors
- LeReS: download weights and place in root repository directory, run as described by the authors
- MiDaS v3: see below for details

## added fix to memory overflow Leres implementation 

The implementation of running Leres seems to causing a memory leak, as evidenced by continuous errors such as
```
RuntimeError: CUDA out of memory. Tried to allocate 2.56 GiB (GPU 0; 15.90 GiB total capacity; 10.38 GiB already allocated; 1.83 GiB free; 2.99 GiB cached)
```
The solution to this is catching and checking errors, as well as doing a memory collection in python to maximize the available memory before retrying the model again. the options tend to be very limited since estimating patches is a very memory intensive task, but this is a an easy fix to this problem.
```py
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
```

## Adding a depth model (MiDaS v3) to the code baseline
Demo code for MiDaS v3 was adjusted from the MiDaS repository, such that it runs a whole folder of images in `run_midasV3.py`.
It automatically downloads the right weights from torchhub and the right transformation for the original estimation.
In order to add the model to the whole pipeline several changes need to be made:
- adding the option to select a model via command line interface `--depthnet` which needs to be changed in `run.py` and `pix2pix/options/base_options.py`
- add model loading when new depthnet is selected in `run.py`
    ```py    
      # MiDasV3
        elif option.depthNet == 3:
            # model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
            model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
            # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    
            global midasmodelv3
            midasmodelv3 = torch.hub.load("intel-isl/MiDaS", model_type)
            midasmodelv3.to(device)
            midasmodelv3.eval()
    ```
- add inference method to `run.py`, in our case `midasestimateV3` which was similar to the inference of MiDaS v2
- add redirection to the `singleestimate` method in `run.py`
    ```py    
      elif net_type == 3:
          return estimatemidasV3(img, msize)
    ```

## Testing SGR, MiDaS v2 and MiDaS v3 with the DIODE dataset
Similar to before, just changing the input directory to the rgb image directory of DIODE.

### Pre-processing the ground truth data of DIODE for evaluation
In the `data_processing/convert_npy_depth.py` file we convert the DIODE npy depth data to png such that the evaluation method does not need to be changed.

## Computing R20 for any given image
In the `R20_computation.py`, with code snippets from the main `run.py` file, we were able to compute the optimal high resolution R20 for any given image as input.

## Merging two images
With `merge2pics.py` we extracted the code of using the pix2pix network and we experimented with merging two images of different depth estimation models.

## Using Demo Code of SGR, MiDaS and LeReS (taken from each models repository and adjusted) to run model on it own
The files `midasV3_demo.py` and `SGR_demo.py` represent the demo files for each network.

The files `run_leres.py` and `run_midas.py` are adjusted demos to run a whole input directory given as command line argument and saving to a output directory.

## Evaluation script modifications
In the case of LeReS, depth is estimated instead of disparity, therefore we swap the two variables when LeReS is evaluated by adding:

```Matlab
if dataset_disp_esttype
    temp = estimate_disp;
    estimate_disp = estimate_depth;
    estimate_depth = temp;
end
```

Moreover, a simple check was added to skip pictures not present in the estimation path.

## Appendix
We also generated 12 megapixel depth maps from images taken by a smartphone using the `run.py` file and different depth models.

The file `run_colab.ipynb` was used to run our benchmarks in Google Colab where we saved the resulting output images to Drive.

In the `run.py` file the following lines were added to skip images that have been processed already (existing in output directory).

```py
if os.path.isfile(os.path.join(result_dir, images.name + '.png')):
    print('skipping image', image_ind, ':', images.name)
    continue
```


Note: Our resulting 100 estimates for each dataset and model are not uploaded to the repository.

Some output images of SGR were in RGB format even if they appeared to be in grayscale. They were converted using `data_processing/rgb2gray_files.py`.