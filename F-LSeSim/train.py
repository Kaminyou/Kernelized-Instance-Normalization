"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
from collections import defaultdict

from torchvision.utils import save_image

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions


# from util.visualizer import Visualizer
def reverse_image_normalize(img, mean=0.5, std=0.5):
    return img * std + mean


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print("The number of training images = %d" % dataset_size)

    model = create_model(
        opt, normalization_mode="in"
    )  # create a model given opt.model and other options

    total_iters = 0  # the total number of training iterations

    os.makedirs(os.path.join("./checkpoints/", opt.name, "test"), exist_ok=True)
    for epoch in range(
        opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1
    ):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        out = defaultdict(int)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            print(f"[Epoch {epoch}][Iter {i}] Processing ...", end="\r")
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()
                model.print_networks(True)
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if (
                total_iters % opt.display_freq == 0
            ):  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                results = model.get_current_visuals()

                for img_name, img in results.items():
                    save_image(
                        reverse_image_normalize(img),
                        os.path.join(
                            "./checkpoints/",
                            opt.name,
                            "test",
                            f"{epoch}_{img_name}_{i}.png",
                        ),
                    )

                for k, v in out.items():
                    out[k] /= opt.display_freq

                print(f"[Epoch {epoch}][Iter {i}] {out}", flush=True)
                for k, v in out.items():
                    out[k] = 0

            losses = model.get_current_losses()
            for k, v in losses.items():
                out[k] += v

            if (
                total_iters % opt.save_latest_freq == 0
            ):  # cache our latest model every <save_latest_freq> iterations
                print(
                    "saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                save_suffix = "iter_%d" % total_iters if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if (
            epoch % opt.save_epoch_freq == 0
        ):  # cache our model every <save_epoch_freq> epochs
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)
        )
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
