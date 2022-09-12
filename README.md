# GMM-SLIC-RustDetction
This is a novel rust segmentation approach based on the Gaussian mixture model (GMM) and SLIC superpixel segmentation.

Training code, testing code, and testing images are provided. For the training images and the pre-trained models (including `GMMmodel_RGB.pkl`, `GMMmodel_HSV.pkl`, and `GMMmodel_combine.pkl`), please contact yangzhixinluo@link.cuhk.edu.cn.

## Citation
Please cite 

## Preparation
Install required dependencies:
```sh
pip install -r requirements.txt
```
Make a new directory `./Output` under the working directory for the output images that will be generated:
```sh
cd GMM-SLIC-RustDetction # Comes to the working directory that you put this project in
mkdir Output
```

## Usage of training/testing code
The following files are used to train the GMM model:
* `Train_GMM_RGB.py`: Training code for GMM model using RGB features
* `Train_GMM_HSV.py`: Training code for GMM model using HSV features
* `Train_GMM_Combine.py`: Training code for GMM model using RGB+HSV features

The following files are used to test the GMM model with SLIC superpixel segmentation, and generate output images:
* `Test_RGB_SLIC.py`: Testing code with SLIC segmentation for GMM model using RGB features
* `Test_HSV_SLIC.py`: Testing code with SLIC segmentation for GMM model using HSV features
* `Test_Combine_SLIC.py`: Testing code with SLIC segmentation for GMM model using RGB+HSV features

## Training
Train GMM model using RGB features:
```sh
python Train_GMM_RGB.py --train_path=<your-training-images-directory-path>
```
Train GMM model using HSV features:
```sh
python Train_GMM_HSV.py --train_path=<your-training-images-directory-path>
```
Train GMM model using RGB+HSV features:
```sh
python Train_GMM_Combine.py --train_path=<your-training-images-directory-path>
```
Example:
```sh
python Train_GMM_HSV.py --train_path="./train_images"
```

## Testing
Test with SLIC segmentation for GMM model using RGB features:
```sh
python Test_RGB_SLIC.py --test_img_path=<your-testing-image-path>
```
Test with SLIC segmentation for GMM model using HSV features:
```sh
python Test_HSV_SLIC.py --test_img_path=<your-testing-image-path>
```
Test with SLIC segmentation for GMM model using RGB+HSV features:
```sh
python Test_Combine_SLIC.py --test_img_path=<your-testing-image-path>
```
Example:
```sh
python Test_HSV_SLIC.py --test_img_path="./test_images/test1.png"
```

## Output
### Training output
The trained model `.pkl` file (according to which training file you ran) will be generated under `./Model` in the working directory:
* `GMMmodel_RGB.pkl`: The trained model using RGB features
* `GMMmodel_HSV.pkl`: The trained model using HSV features
* `GMMmodel_combine.pkl`: The trained model using RGB+HSV features

### Testing output
All the output images will be generated under `./Output` in the working directory, including (take running `Test_HSV_SLIC.py` as example):
* `HeatMap_GMM_HSV.png`: The heat image (visualized probability) predicted by the GMM model
* `SLIC_Black_HSV.jpg`: The SLIC superpixel segmentation output (black background) of the original input image
* `SLIC_White_HSV.jpg`: The SLIC superpixel segmentation output (white background) of the original input image
* `SLIC_HSV.jpg`: The SLIC superpixel segmentation output (original background) of the original input image
* `Final_Img_HSV.jpg`: The final segmentation output with rust detected in red and original background
* `Binary_Img_HSV.jpg`: The binary segmentation output with rust detected in red and background in white

## Remarks
1. We have supported the image formats including `PNG`, `JPG`, `JPEG`, `png`, `jpg`, `jpeg` for the training and testing images.
2. If your environment does not support GUI display (like when running on remote servers), just comment out all the lines of `cv2.imshow()`, `cv2.waitKey(0)`, `cv2.destroyAllWindows()`.
3. Notice that do not comment out the lines of `cv2.imwrite()`, otherwise you will not get the output images under `./Output` in the working directory.
