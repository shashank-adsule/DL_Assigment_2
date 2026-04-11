from .layers       import CustomDropout, IoULoss
from .vgg11        import VGG11Encoder, VGG11          # autograder uses VGG11Encoder
from .localization import LocalizationModel
from .segmentation import UNetVGG11, DiceCELoss
from .multitask    import MultiTaskVGG11, MultiTaskLoss, MultiTaskPerceptionModel

__all__ = [
    "CustomDropout",
    "IoULoss",
    "VGG11Encoder",
    "VGG11",
    "LocalizationModel",
    "UNetVGG11",
    "DiceCELoss",
    "MultiTaskVGG11",
    "MultiTaskPerceptionModel",    # autograder uses this name
    "MultiTaskLoss",
]