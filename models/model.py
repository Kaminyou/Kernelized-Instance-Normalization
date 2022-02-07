from models.cut import ContrastiveModel
from models.cyclegan import CycleGanModel


def get_model(config, model_name="CUT", normalization="in"):
    if model_name == "CUT":
        model = ContrastiveModel(config, normalization=normalization)
    elif model_name == "cycleGAN":
        model = CycleGanModel(config, normalization=normalization)
    else:
        raise NotImplementedError
    return model
