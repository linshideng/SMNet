# SMNet
SMNet: Synchronous Multi-scale Low Light EnhancementNetwork with Local and Global Concern.

by SHIDENG LIN, FAN TANG, WEIMING DONG, XINGJIA PAN and CHANGSHENG XU.

The SMNet has good performance to enhance low-light images.
## Abstract
## Method
(framework)

## Requirements
SMNet is built by:
- python 3.7.3
- pytorch 1.1.0
- scikit-image   0.14.5  
- scipy   1.1.0     
- opencv-python    4.1.1.26   
- pillow   5.2.0 

## Usage
Tips:
-  The h/w resolution of all input images shall be divided by 4, such as 600*400.
-  We provide a few test images in `./dataset`.
### Testing
To test the pre-trained model on LOL dataset, just run:
```
bash test.sh
```

or run:

```
python test.py  --test_folder  path_to_images  --output save_images_here  --modelfile pretrained_model --modeltype lol
```
Example:
```
python test.py  --test_folder  ./datasets/LOL/test/low  --output  ./output_test  --modelfile ./model_LOL.pth --modeltype LOL
```
- You can change the `--test_folder` to test your own dataset.
- You can use the pretrained model training on Adobe-MIT FiveK dataset by `--modelfile ./model_FIVEK.pth --modeltype FIVEK`
### Training
Before training  on your own dataset, you should place images like 
- your_dataset
  - train
    - high
    - low
  - test
    - high
    - low

moreover, you should use the `vgg16.pth` for perceptual loss. The file has been uploaded to [Baidu Netdisk](https://pan.baidu.com/s/1QIV50-mN_3NpMg2BYbKM7Q)(code: hzfs) and [Goole Drive](https://drive.google.com/file/d/1b1odcQXTSJSnWZBu3PWPDei9zYTFIxSu/view?usp=sharing). And please change the **absolute path** in line 56 of `lib/utils.py`.  

`train` `low`/`high` includes low-light images and their ground truth for training, while `test` `low`/`high ` includes low-light images and their ground truth for testing.

To train the model, just run:

```
bash train.sh
```
or run:
```
python train.py --trainset  path_to_trainset  --testset path_to_testset  --output  save_inter_images
```
Example:
```
python train.py --trainset  ./datasets/LOL/train  --testset  ./datasets/LOL/test  --output  ./output
```
