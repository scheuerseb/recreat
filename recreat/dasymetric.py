import numpy as np
import numpy.ma as ma

class DasymetricMapping:



    samples = None

    def __init__(self):
        self.samples = {}


    def sample_class(self, rst_residential_count: np.ndarray, rst_population: np.ndarray, threshold_count: int, min_sample_size: int = 3):                

        mask_residential = rst_residential_count >= threshold_count
        
        masked_residential = ma.array(rst_residential_count, mask=mask_residential)
        
        #data_residential = np.extract(mask_residential, rst_residential_count)


        print(data_residential.shape)

