# URUST: Ultra-resolution unpaired stain transformation via Kernelized Instance Normalization
...

### Knowledgement
Please refer to the official implementation [here](https://github.com/taesungp/contrastive-unpaired-translation). This code is a simplified version revised from [wilbertcaine's implementation](https://github.com/wilbertcaine/CUT).

## Usage (Training)
Suppose you would like to transfer image from source domain `X` to the target domain `Y`.
1. Please split your data `X` into training and testing set. Namely, split `X` into `X_train` and `X_test`.
2. Put `X_train` into `./data/trainX`. 
3. Put `X_test` into `./data/testX`. 
4. Put `Y` into `./data/trainY`.
5. Modify `./config.yaml` if you would like to adjust some setting, or just keep the default setting.
6. Execute `python3 train.py`.
7. Some transfered examples will be generated during training. Please check the `./experiments/$experiment_name/train/` folder.

## Usage (Inference)
```
python3 inference.py
```
The transfered images will be stored in `./experiments/$experiment_name/test/` folder.

## Environment
- Python 3.8.6
- All the required packages are listed in the `requirements.txt`.