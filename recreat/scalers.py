###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################

import rasterio
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from enum import Enum
import os
from .base import RecreatBase

class ScalingMethod(Enum):
    MinMaxNormalization = 'min-max'
    RobustScaling = 'robust'

class Scaler(RecreatBase):

    @staticmethod
    def scale(filename: str, mode : ScalingMethod) -> None:
        RecreatBase.printStepInfo(msg = "SCALING...")
        # read input and input properties
        rst = rasterio.open(filename)
        rst_meta = rst.meta.copy()

        # get data any apply scaling
        mtx = rst.read(1)
        mtx = Scaler.apply_scaling(mtx, mode)
        
        # export result
        out_filename = f"{os.path.dirname(filename)}/{mode.value}_scaled_{os.path.basename(filename)}"
        with rasterio.open(out_filename, "w", **rst_meta) as dest:
            dest.write(mtx, 1)

        RecreatBase.printStepCompleteInfo()


    @staticmethod
    def apply_scaling(mtx : np.ndarray, mode: ScalingMethod) -> np.ndarray:
        
        # store original mtx shape
        orig_shape = mtx.shape
        
        # apply scaling on reshaped mtx
        scaler = None
        if mode is ScalingMethod.MinMaxNormalization:
            scaler = MinMaxScaler()
        elif mode is ScalingMethod.RobustScaling:
            scaler = RobustScaler()        
        mtx = scaler.fit_transform(mtx.reshape([-1,1]))

        # return result reshaped to original shape
        return mtx.reshape(orig_shape)


        
