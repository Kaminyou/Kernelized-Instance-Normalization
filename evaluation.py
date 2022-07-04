import os

exps = ["lung_lesion"]
models = ["cycleGAN", "CUT", "LSeSim"]
model_epochs = [60, "latest", 100]
norm_types = ["in", "tin", "kin"]
kernel_types = [
    None,
    None,
    [
        "constant_1",
        "gaussian_1",
        "constant_3",
        "gaussian_3",
        "constant_5",
        "gaussian_5",
    ],
]

for exp in exps:
    for i, model in enumerate(models):
        model_epoch = model_epochs[i]
        for j, norm_type in enumerate(norm_types):
            kernel_type = kernel_types[j]
            if kernel_type is None:
                command = f"python3 metric.py --exp_name {model}_{model_epoch}_{norm_type} --path-A ./experiments/{exp}/{exp}_{model}/test/{norm_type}/{model_epoch}/ --path-B ./data/lung_lesion/testX --blank_patches_list ./data/lung_lesion/testX/blank_patches_list.csv >> FID.out"
                os.system(command)
            else:
                for kernel in kernel_type:
                    command = f"python3 metric.py --exp_name {model}_{model_epoch}_{norm_type}_{kernel} --path-A ./experiments/{exp}/{exp}_{model}/test/{norm_type}/{model_epoch}/{kernel}/ --path-B ./data/lung_lesion/testX --blank_patches_list ./data/lung_lesion/testX/blank_patches_list.csv >> FID.out"
                    os.system(command)
