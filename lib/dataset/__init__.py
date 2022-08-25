from ._360cc import _360CC
from ._own import _OWN
from .pretrain import pretrain
from .mysynthtext import mysynthtext

def get_dataset(config):

    if config.DATASET.DATASET == "360CC":
        return _360CC
    elif config.DATASET.DATASET == "icdar2015":
        return _OWN
    elif config.DATASET.DATASET == "pretrain":
        return pretrain
    elif config.DATASET.DATASET == "mysynthtext":
        return mysynthtext
    else:
        raise NotImplemented()