from models.cut import ContrastiveModel
from models.cyclegan import CycleGanModel


def get_model(config, model_name="CUT", norm_cfg=None, isTrain=True):
    if model_name == "CUT":
        model = ContrastiveModel(config, norm_cfg=norm_cfg)
    elif model_name == "cycleGAN":
        model = CycleGanModel(config, norm_cfg=norm_cfg)
    elif model_name == "LSeSim":
        print("Please use the scripts prepared in the F-LSeSim folder")
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model
