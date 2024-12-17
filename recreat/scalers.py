###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################

import rasterio
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer
from enum import Enum
import os
from .base import RecreatBase

class ScalingMethod(Enum):
    MinMaxNormalization = 'min-max'
    RobustScaling = 'robust'
    QuantileTransform = 'qtransform'

class Scaler(RecreatBase):

    @staticmethod
    def scale(in_filename: str, clump_filename: str, out_filename: str, mode : ScalingMethod, scaler = None, inverse: bool = False) -> None:
        
        RecreatBase.printStepInfo(msg = f"Scaling {in_filename}")
        
        # read input and input properties
        rst = rasterio.open(in_filename)
        rst_meta = rst.meta.copy()
        rst_clumps = rasterio.open(clump_filename).read(1)
        mask = rst_clumps > 0

        # get data any apply scaling
        mtx_in = np.extract(mask, rst.read(1))
        mtx_in = Scaler.apply_scaling(mtx_in, mode=mode, scaler=scaler, inverse=inverse)
                
        mtx_out = np.zeros(rst_clumps.shape, dtype=np.float32)
        np.place(mtx_out, mask, mtx_in)

        # export result
        # out_filename = f"{os.path.dirname(filename)}/scaled_{os.path.basename(filename)}"
        with rasterio.open(out_filename, "w", **rst_meta) as dest:
            dest.write(mtx_out, 1)

        RecreatBase.printStepCompleteInfo()


    @staticmethod
    def apply_scaling(mtx : np.ndarray, mode: ScalingMethod, scaler = None, inverse: bool = False) -> np.ndarray:
        
        # store original mtx shape
        orig_shape = mtx.shape
        
        # apply scaling on reshaped mtx
        if scaler is None:
            if mode is ScalingMethod.MinMaxNormalization:
                scaler = MinMaxScaler()
            elif mode is ScalingMethod.RobustScaling:
                scaler = RobustScaler()        
            elif mode is ScalingMethod.QuantileTransform:
                scaler = QuantileTransformer()

        mtx = scaler.fit_transform(mtx.reshape([-1,1]))
        if inverse:
            np.subtract(1, mtx, out=mtx)

        # return result reshaped to original shape
        return mtx.reshape(orig_shape)


        
