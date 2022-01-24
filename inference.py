import os
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.cut import ContrastiveModel
from utils.dataset import XYDataset
from utils.util import read_yaml_config, test_transforms, reverse_image_normalize


def main():
    config = read_yaml_config("./config.yaml")

    model = ContrastiveModel(config)

    val_dataset = XInferenceDataset(root_X=config["INFERENCE_SETTING"]["TEST_DIR_X"], transform=test_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    model.load_networks(config["INFERENCE_SETTING"]["MODEL_VERSION"])

    for idx, data in enumerate(val_loader):
        print(f"Processing {idx}", end="\r")

        X, X_path = data
        Y_fake = model.inference(X)

        if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
            save_image(reverse_image_normalize(X), os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test", f"{Path(X_path[0]).stem}_X_{idx}.png"))
        
        save_image(reverse_image_normalize(Y_fake), os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test", f"{Path(X_path[0]).stem}_Y_fake_{idx}.png"))

if __name__ == "__main__":
    main()
