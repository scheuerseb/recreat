###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################

# opencv-python
import cv2 as cv
import rasterio
import numpy as np
from enum import Enum
from typing import Tuple, List, Callable, Dict
# colorama
from colorama import init as colorama_init
from colorama import Fore, Back, Style, just_fix_windows_console
just_fix_windows_console()



class recreational_dimension(Enum):
    DIVERSITY = 'diversity'
    SUPPLY = 'total-supply'
    FLOW = 'flow'
    DEMAND = 'demand'
    COST = 'cost'

class clustering:

    @staticmethod    
    def _get_path_for_dimension(dimension: recreational_dimension) -> np.ndarray:
        file_path = None

        if dimension is recreational_dimension.SUPPLY:
            file_path = 'INDICATORS/scaled_non_weighted_avg_totalsupply.tif'
        elif dimension is recreational_dimension.DIVERSITY:
            file_path = 'INDICATORS/scaled_non_weighted_avg_diversity.tif'
        elif dimension is recreational_dimension.COST:
            file_path = 'INDICATORS/scaled_non_weighted_avg_cost.tif'
        elif dimension is recreational_dimension.FLOW:
            pass

        return file_path

    @staticmethod
    def kmeans(data_path: str, root_path: str, k: int, dimensions: List[recreational_dimension] = [recreational_dimension.SUPPLY, recreational_dimension.DIVERSITY, recreational_dimension.COST], attempts: int = 10) -> None:
        """Make a map of clusters of recreational potential, derived from scaled, non-weighted averaged indicators 

        :param data_path: Data-path to use.
        :type data_path: str
        :param root_path: Root-path to use.
        :type root_path: str
        :param k: Number of clusters.
        :type k: int
        :param dimensions: Dimensions of landscape recreational potential to be included in the clustering, defaults to [recreational_dimension.TOTAL_SUPPLY, recreational_dimension.DIVERSITY, recreational_dimension.COST].
        :type dimensions: List[recreational_dimension], optional
        :param attempts: Number of attempts, defaults to 10.
        :type attempts: int, optional
        """
        if not dimensions:            
            return

        assessed_dimensions = [f.value for f in dimensions]
        print(Fore.WHITE + Style.DIM + ", ".join(assessed_dimensions) + Style.RESET_ALL)
        
        # include all cells where clumps > 0
        src_clumps = rasterio.open(f"{data_path}/{root_path}/MASKS/clumps.tif")
        rst_clumps = src_clumps.read(1)
        clumps_mask = rst_clumps > 0
        src_meta = src_clumps.meta
        del rst_clumps

        indata = {
            d: np.extract(clumps_mask, rasterio.open( f"{data_path}/{root_path}/{self._get_path_for_dimension(d)}").read(1)).reshape(-1, 1)
            for d in dimensions
        }

        cluster_input = np.hstack(tuple(indata.values()))
        cluster_input = np.float32(cluster_input) # to be sure

        # Apply KMeans
        print("STARTING CLUSTERING")
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv.kmeans(cluster_input, k, None, criteria, attempts, flags)
        print(Fore.YELLOW + Style.BRIGHT + f"COMPACTNESS={compactness}" + Style.RESET_ALL)

        # we need an array of zeros with the same shape as lsm
        final_labels = np.zeros(src_clumps.shape, dtype=np.int32)
        np.place(final_labels, clumps_mask, labels + 1)

        with rasterio.open(f"{data_path}/labels_{k}.tif", "w", **src_meta) as dest:
            dest.write(final_labels.reshape(src_clumps.shape), 1)