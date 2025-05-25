from . import _types

from .engine import train_one_epoch

from . import data
from . import optim
from . import layers, blocks
from . import miscs
from . import transforms


def _initialization():
    import numpy

    numpy.set_printoptions(precision=4)
