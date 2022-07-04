import os

generated_whole_slide_roots = [
    "/home/vincentwu/URUST/experiments/lung_lesion/lung_lesion_cycleGAN/test/",
    "/home/vincentwu/URUST/experiments/lung_lesion/lung_lesion_CUT/test/",
    "/home/vincentwu/URUST/experiments/lung_lesion/lung_lesion_LSeSim/test/",
]

reference_image = "/home/vincentwu/URUST/data/ANHIR2019/dataset_medium/lung-lesion_1/scale-50pc/29-041-Izd2-w35-He-les1.jpg"
for generated_whole_slide_root in generated_whole_slide_roots:
    print(f"==============================================================")
    print(f"{generated_whole_slide_root}")
    print(f"==============================================================")
    for image in os.listdir(generated_whole_slide_root):
        if (
            not os.path.isdir(os.path.join(generated_whole_slide_root, image))
            and image.split(".")[1] == "png"
        ):
            print(f"================ {image} ================")
            command = f"python3 metric_whole_image_with_ref.py --image_A_path {reference_image} --image_B_path {os.path.join(generated_whole_slide_root, image)}"
            os.system(command)
            command = f"python3 metric_whole_image_no_ref.py --path {os.path.join(generated_whole_slide_root, image)} --save_grad"
            os.system(command)
