# Fast neural style transform

## This is the repository of the fast neural style transform.

## Function description

`Train.py` is the implementation for training transform model.

`Eval.py` is the implementation for evaluating the pre-trained transform model.

`webcam.py` is an implementation that applies the pre-trained model to a webcam to achieve real-time style image transform.

----

## Dataset

### COCO 2014 training dataset

Full (13GB): http://images.cocodataset.org/zips/train2014.zip

Skimmed down 32k images (5GB): https://mega.nz/file/V88Anbqa#HMeRYGhuc3llG4vjK4lY378T3PM4shEOXuO7L1_uIzc

----

## Requirements

`Module` : Pytorch, torchvision, PIL, OpenCV, matplotlib, tqdm

`optional` : Ideally a CUDA device to train and use realtime webcam style transform

----

## User Instructions

### You should only need to change the global parameters

## `train.py`


```python

# Global params
DATASET_DIR = "E:/.../dataset/train"
MODEL_FOLDER = "model"
STYLE_IMG = "monet_3"
TEST_IMG = "hsuan"
STYLE_SCALE = 1

CONTINUE = None

EPOCHS = 1
BATCH_SIZE = 4
TRAIN_IMG_SIZE = 128
LR = 1e-3
CONTENT_WEIGHT = 1e0
STYLE_WEIGHT = 1e6
CHKPT_FREQ = 0
TEST_FREQ = 500

```


### DATASET_DIR : Dataset directory `(string)`

PyTorch dataloader requires the images be located in a subfolder of the directory you entered in.


### MODEL_FOLDER : Your main model save folder `(string)`

Default is "model", which will automatically create a folder called "model" in the main folder
Program will then create style sub-folders in the folder to save .pth files.

### STYLE_IMG : filename of the style reference image `(string)`

Style images should be located inside "images/2-style/" folder.
No need to add file format(".jpg") after the filename.
This string is also used to create a subfolder named after the style image inside the main model folder, your model and checkpoint .pth files will be saved inside said folder.
		
### TEST_IMG : filename of the test image `(string)`

Test images should be located inside "images/1-input" folder.
No need to add file format(".jpg") after the filename.
Program will create a subfolder named "test images" inside the MODEL_SAVE_FOLDER, and save network test image outputs every x iterations, which is set with TEST_FREQ.
Change TEST_SAVE_FOLDER to change where the test images will save to.
		
### STYLE_SCALE : `(float)`

Change the scale of your style image while training. Affects the output.
A 512x512 sized image will be scaled to (512*scale)x(512*scale).
		
### CONTINUE `(string)`

Use this function if you want to continue training from a previous session.Model location should be located inside MODEL_SAVE_FOLDER.
Set to None if you do not need this function.
		
### EPOCHS, BATCH_SIZE, TRAIN_IMG_SIZE, LR : should be self explanitory

### CONTENT_WEIGHT, STYLE_WEIGHT 

Change these values to change how your output image looks.Higher content weight, image will look more like the original image.Higher style weight, image will look more like the style image.
default values are already set to ideal numbers, but feel free to change them!Recommend leaving CONTENT_WEIGHT alone, and change only the STYLE_WEIGHT instead!
		
### CHKPT_FREQ : checkpoint frequency `(int)`

Saves a checkpoint .pth model every x interations.
Set to 0 to disable
		
### TEST_FREQ : test image output frequency `(int)`

Saves a test output image to TEST_SAVE_FOLDER every x iterations
Set to 0 to disable

----
## `Eval.py`

```python
# Global params
MODEL_NAME = "rain_512_epoch_1_cw_1.0_sw_1000000.0"
STYLE = "rain_512"
TEST_IMG = "hsuan"
TEST_SCALE = 1
OUTPUT_IMG = TEST_IMG + "_" + STYLE

```
### MODEL_NAME : model filename`（string)`
## Models should be located inside MODEL_DIR

No need to add file format (".pth") after the filename
Change MODEL_DIR if you want an absolute path to the model file
		
### STYLE : style image name`（string)`

Used to choose the subfolder inside the main model folder, and also rename the output filename
		
### TEST_IMG : test image filename`（string)`

Test images should be located inside "images/1-input" folder.
No need to add file format (".jpg") after the filename.
Change TEST_IMG_DIR if you want an absolute path to the test image file
	
### TEST_SCALE : test image scale`(float)`

Change the scale of your input image during evaluation. Affects the output.A 512x512 sized image will be scaled to (512*scale)x(512*scale).
		
### OUTPUT_IMG : output image filename`（string)`

Output images will be located inside "images/3-output"
No need to add file format (".jpg") after the filename.
Change OUT_IMG_DIR if you want an absolute path to the output image file

----

## `webcam.py`

```python
# Global Parameters
MODEL_PATH = "model/wave/wave_epoch_2_cw_1.0_sw_1000000.0.pth"
SCALE = 1
```
### MODEL_DIR : absolute or relative path to the model`（string)`

### SCALE : `(float)`
Change the scale of your webcam frame during evaluation. Affects the output.A 512x512 sized frame will be scaled to (512*scale)x(512*scale).