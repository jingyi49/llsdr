__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

import audiotools

audiotools.ml.BaseModel.INTERN += ["vodac.**"]
audiotools.ml.BaseModel.INTERN += ["audiotool.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]
audiotools.ml.BaseModel.EXTERN += ["rich."]
audiotools.ml.BaseModel.EXTERN += ["einops_exts"]


from . import nn
from . import model
from . import utils
from .model import VODAC
from .model import DACFile
