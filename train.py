import argparse
import os
from collections import defaultdict

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.cut import ContrastiveModel
from utils.dataset import XYDataset
from utils.util import read_yaml_config, reverse_image_normalize, transforms


def main():
    parser = argparse.ArgumentParser("Model training")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml", help="Path to the config file.")
    args = parser.parse_args()
    
    config = read_yaml_config(args.config)
    
    model = ContrastiveModel(config)

    dataset = XYDataset(root_X=config["TRAINING_SETTING"]["TRAIN_DIR_X"], root_Y=config["TRAINING_SETTING"]["TRAIN_DIR_Y"], paired=config["TRAINING_SETTING"]['PAIRED_TRAINING'], transform=transforms)
    dataloader = DataLoader(dataset, batch_size=config["TRAINING_SETTING"]["BATCH_SIZE"], shuffle=True, num_workers=config["TRAINING_SETTING"]["NUM_WORKERS"])

    for epoch in range(config["TRAINING_SETTING"]["NUM_EPOCHS"]):
        out = defaultdict(int)

        for idx, data in enumerate(dataloader):
            print(f"[Epoch {epoch}][Iter {idx}] Processing ...", end="\r")
            model.set_input(data)
            model.optimize_parameters()

            if idx % config["TRAINING_SETTING"]["VISUALIZATION_STEP"] == 0 and idx > 0:
                results = model.get_current_visuals()

                for img_name, img in results.items():
                    save_image(reverse_image_normalize(img), os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "train", f"{epoch}_{img_name}_{idx}.png"))
                
                for k, v in out.items():
                    out[k] /= config["TRAINING_SETTING"]["VISUALIZATION_STEP"]

                print(f"[Epoch {epoch}][Iter {idx}] {out}", flush=True)
                for k, v in out.items():
                    out[k] = 0
            
            losses = model.get_current_losses()
            for k, v in losses.items():
                out[k] += v

        model.scheduler_step()
        if epoch % config["TRAINING_SETTING"]["SAVE_MODEL_EPOCH_STEP"] == 0 and config["TRAINING_SETTING"]["SAVE_MODEL"]:
            model.save_networks(epoch)
    
    model.save_networks("latest")

if __name__ == "__main__":
    main()
