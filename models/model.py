from models.cut import ContrastiveModel
from models.cyclegan import CycleGanModel
from models.lsesim import LSeSim


def get_model(config, model_name="CUT", normalization="in", isTrain=True):
    if model_name == "CUT":
        model = ContrastiveModel(config, normalization=normalization)
    elif model_name == "cycleGAN":
        model = CycleGanModel(config, normalization=normalization)
    elif model_name == "LSeSim":
        model = LSeSim(config, normalization=normalization, isTrain=isTrain)
    else:
        raise NotImplementedError
    return model
