# URUST: Ultra-resolution unpaired stain transformation via Kernelized Instance Normalization
We invented a kernelized instance normalization module enabling ultra-resolution unpaired stain transformation

## Environment
- Python 3.8.6
- All the required packages are listed in the `requirements.txt`.

## Quick instruction
A simple example is provided for you here. Or you can jump to the next section to train a model for your own dataset.
### Prepare dataset
A public dataset [ANHIR](https://anhir.grand-challenge.org/Data/) is used in this project. Please first download it from the offical website and put `ANHIR2019/dataset_medium/breast_1/scale-20pc/HE.jpg` and `ANHIR2019/dataset_medium/breast_1/scale-20pc/ER.jpg` in `data/example/` folder. We would like to transfer `HE (domain X)` to `ER (domain Y)`.

### Take a look at the `config.yaml`
The whole pipeline is heavily dependent on the `config.yaml`. Please take a look at the `./data/example/config.yaml` first to understand what are necessary during training and testing process. You can easily train your own model with your own dataset by modifiying the `config.yaml`.

### Preprocessing
1. It is recommended to manually crop a center part from `HE.jpg` and `ER.jpg` first as the main contents are surrounded by a lot of unnecessary blank region, which will increase the training time but make the distribution hard to be learned. 
2. Assume these two images are cropped and place at `./data/example/HE_cropped.jpg` and `./data/example/ER_cropped.jpg`.
3. Execute the following script to crop patches for training and testing.
```
python3 crop_pipeline.py -c ./data/example/config.yaml
```

### Training
1. Train the model
```script
python3 train.py -c ./data/example/config.yaml
```
2. Wait for the model training
- Some transfered examples will be generated during training. Please check the `./experiments/example/train/` folder.

### Inference
As the testing data have been cropped during the first step, we can skip this step here.
```
python3 transfer.py -c config_example.yaml --skip_cropping
```
The output will be in the `./experiments/example/test/` folder.

## Train your own model with your own dataset
1. Create a folder in `./data/`
2. Put a `config.yaml` in `./data/$your_folder/`
3. Prepare images (domain X) and images in (domain Y) in `./data/$your_folder/`.
4. Crop those images into patches.
- If there is only one image in each domain
```
python3 crop_pipeline.py -c ./data/$your_folder/config.yaml
```
- **Multiple images belong to one domain:** you should use `crop.py` to crop each image and save those patches in the same folder (`trainX`, `trainY`)
```
python3 crop.py -i ./data/$your_folder/$image_a -o ./data/$your_folder/trainX/ --thumbnail_output ./data/$your_folder/trainX/
python3 crop.py -i ./data/$your_folder/$image_b -o ./data/$your_folder/trainX/ --thumbnail_output ./data/$your_folder/trainX/
...
```
- **Multiple images belong to one domain:**: for the testing data, it is recommended to seperate patches belong to different image in different folder.
```
python3 crop.py -i ./data/$your_folder/$test_a -o ./data/$your_folder/$test_a/ --stride 512 --thumbnail_output ./data/example/$test_a/
python3 crop.py -i ./data/$your_folder/$test_b -o ./data/$your_folder/$test_b/ --stride 512 --thumbnail_output ./data/example/$test_b/
...
```
5. Modify `TRAINING_SETTING` section in `./data/$your_folder/config.yaml`, especially the `TRAIN_DIR_X` and `TRAIN_DIR_Y`.
6. Train the model
```
python3 train.py -c ./data/$your_folder/config.yaml
```
7. **If you have only one image requires inference:** modify `INFERENCE_SETTING` section in `./data/$your_folder/config.yaml`, especially the `TEST_X` and `TEST_DIR_X`. Then,
```
python3 transfer.py -c config_example.yaml --skip_cropping
```
8. **If you have only many images requires inference:** assume you have finishing cropping each testing image in separated folder. Please modify `TEST_X` and `TEST_DIR_X` in the `INFERENCE_SETTING` section and execute the following script for each image.
```
python3 transfer.py -c config_example.yaml --skip_cropping
```


## Knowledgement
Besides our novel kernelized instance normalizatio module, we use [Contrastive Unpaired Translation](https://link.springer.com/chapter/10.1007/978-3-030-58545-7_19) as our backbone. Please refer to the official implementation [here](https://github.com/taesungp/contrastive-unpaired-translation). This code is a simplified version revised from [wilbertcaine's implementation](https://github.com/wilbertcaine/CUT).