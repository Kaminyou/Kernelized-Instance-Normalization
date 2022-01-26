# URUST: Ultra-resolution unpaired stain transformation via Kernelized Instance Normalization
We invented a kernelized instance normalization module enabling ultra-resolution unpaired stain transformation

## Usage (Training)
A public dataset [ANHIR](https://anhir.grand-challenge.org/Data/) is used in this project. Please first download it from the offical website and put `ANHIR2019/dataset_medium/breast_1/scale-20pc/HE.jpg` and `ANHIR2019/dataset_medium/breast_1/scale-20pc/ER.jpg` in `data/example/` folder. We would like to transfer `HE (domain X)` to `ER (domain Y)`.

1. It is recommended to manually crop a center part from `HE.jpg` and `ER.jpg` first as the main contents are surrounded by a lot of unnecessary blank region, which will increase the training time but make the distribution hard to be learned. 
2. Crop `HE.jpg` and `ER.jpg` into patches.
```script
python3 crop.py -i ./data/example/HE_cropped.jpg -o ./data/example/trainX/ --thumbnail_output ./data/example/trainX/
python3 crop.py -i ./data/example/ER_cropped.jpg -o ./data/example/trainY/ --thumbnail_output ./data/example/trainY/
```
3. Train the model
```script
python3 train.py --config config_example.yaml
```
4. Wait for the model training
- Some transfered examples will be generated during training. Please check the `experiments/example/train/` folder.

## Usage (Inference)
1. Do inference for the images in `INFERENCE_SETTING.TEST_DIR_X` folder specified in the `config` file.
```
python3 inference.py --config config_example.yaml
```
- The transfered images will be stored in `./experiments/$experiment_name/test/` folder.
2. Then, you can combine all the images into one ultra-resolution image.
```
python3 combine.py --config config_example.yaml
```
- You can also specify `height` and `width` to resize the transferred image to match the original one.
```
python3 combine.py --config config_example.yaml --resize_h $H --resize_w $W
```

## Usage (Transfer by one step)
```
python3 transfer.py -c config_example.yaml -i ./data/example/HE_cropped.jpg -o ./data/example/testX/ 
```

## Environment
- Python 3.8.6
- All the required packages are listed in the `requirements.txt`.

## Knowledgement
Besides our novel kernelized instance normalizatio module, we use [Contrastive Unpaired Translation](https://link.springer.com/chapter/10.1007/978-3-030-58545-7_19) as our backbone. Please refer to the official implementation [here](https://github.com/taesungp/contrastive-unpaired-translation). This code is a simplified version revised from [wilbertcaine's implementation](https://github.com/wilbertcaine/CUT).