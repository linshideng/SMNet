# SMNet
SMNet: Synchronous Multi-scale Low Light EnhancementNetwork with Local and Global Concern
## Abstract
## Method
(framework)

## Requirements
- python 3.7.3
- pytor 1.1.0
- scikit-image   0.14.5  
- scipy   1.1.0     
- opencv-python    4.1.1.26   
- pillow   5.2.0 

## Usage
Tips: The h/w resolution of all input images shall be divided by 4.
### Testing
To test the pre-trained model on your own images, run:

```
ptyhon test.py --test_folder path_to_images  --output save_images_here --modelfile pretrained_model 
```
Example:
```
ptyhon test.py --test_folder ./datasets/LOL/test/lowpath_to_images  --output ./output_test --modelfile ./model.pth
```

### Training
Before training  on your own dataset, you should place images like 
- your_image_path
  - train
    - high
    - low
  - test
    - high
    - low


where low-light images are placed in low, ground truth are placed in high.

To train the model, run:
```
python train.py --trainset path_to_trainset --testset path_to_testset --output save_inter_images
```
Example:
```
python train.py --trainset ./datasets/LOL/train --testset ./datasets/LOL/test --output ./output
```
