###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################
import os
from os import listdir
from os.path import isfile, join
import ctypes
import platform
import concurrent.futures
from skimage import data, util, measure
import pandas as pd  
import multiprocessing as mp

cv_max_pixel_count = pow(2,40).__str__()
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = cv_max_pixel_count

# opencv-python
import cv2 as cv

import uuid
# colorama
from colorama import init as colorama_init
from colorama import Fore, Back, Style, just_fix_windows_console
just_fix_windows_console()
# rich
from rich.progress import Progress, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn, BarColumn
# contextlib
from contextlib import nullcontext
# rasterio
import rasterio
# distancerasters
import distancerasters as dr
# xarray-spatial
import xarray as xr 
from xrspatial import proximity
#scipy
from scipy import ndimage
from scipy import LowLevelCallable
# numpy
import numpy as np
# scikit-image
from skimage.draw import disk
# scikit-learn
from sklearn.preprocessing import MinMaxScaler
# typing
from typing import Tuple, List, Callable, Dict
from enum import Enum
from string import Template

from .transformations import Transformations
from .disaggregation import SimpleAreaWeightedEngine, DasymetricMappingEngine, DisaggregationMethod
from .exceptions import MethodNotImplemented
from .base import RecreatBase

class Contiguity(Enum):
    Queen = 0
    Rook = 1

class Recreat(RecreatBase):

    # store params
    # define relevant recreation patch and edge classes, cost thresholds, etc.
    lu_classes_recreation_edge = []
    lu_classes_recreation_patch = []
    lu_classes_builtup = []
    cost_thresholds = []
    verbose_reporting = True
    
    # shared library
    clib = None 

    # default values
    nodata_value = -9999

    # reference to key datasets and objects re-used across the implementation
    land_use_map_reader = None
    land_use_map_matrix = None
    clumps_matrix = None
    clumps_nodata_mask = None
   

    def __init__(self, data_path: str):        
        os.system('cls' if os.name == 'nt' else 'clear')

        if not os.path.exists(data_path) and os.access(data_path, os.W_OK):
            print(f"{Fore.RED}Error: data_path not found.{Style.RESET_ALL}")
            raise FileNotFoundError()

        else:
            super().__init__(data_path=data_path, root_path=None)
            print(Fore.WHITE + Style.BRIGHT + "recreat landscape recreational potential (C) 2024, Sebastian Scheuer" + Style.RESET_ALL)
            self.py_path = os.path.dirname(__file__)

            # determine extension module to be used depending on platform
            if platform.system() == "Windows":
                extension_filename = 'LowLevelGenericFilters.dll'
            else:
                extension_filename = 'LowLevelGenericFilters.so'

            clib_path = os.path.join(self.py_path, extension_filename)
            print(Fore.WHITE + Style.DIM + "Using low-level callable shared libary " + clib_path + Style.RESET_ALL)
            self.clib = ctypes.cdll.LoadLibrary(clib_path)

    def __del__(self):        
        print("BYE BYE FROM v2.")


    def make_environment(self) -> None:
        # create directories, if needed
        dirs_required = ['DEMAND', 'MASKS', 'SUPPLY', 'INDICATORS', 'FLOWS', 'FLOWS_DF', 'CLUMPS_LU', 'PROXIMITY', 'COSTS', 'DIVERSITY', 'BASE']
        for d in dirs_required:
            current_path = f"{self.data_path}/{self.root_path}/{d}"
            if not os.path.exists(current_path):
                os.makedirs(current_path)


    def set_params(self, param_name: str, param_value: any) -> None:
        """Set model parameters.

        :param param_name: The model parameter to set/update.
        :type param_name: str
        :param param_value: The model parameter value to set.
        :type param_value: any
        """
        if param_name == 'classes.edge':
            self.lu_classes_recreation_edge = param_value
        elif param_name == 'classes.patch':
            self.lu_classes_recreation_patch = param_value
        elif param_name == 'classes.builtup':
            self.lu_classes_builtup = param_value
        elif param_name == 'costs':
            self.cost_thresholds = param_value    
        elif param_name == 'verbose-reporting':
            self.verbose_reporting = param_value
        elif param_name == 'nodata-value':
            self.nodata_value = param_value

#region Helper functions

    def _get_file(self, filename, nodata_values: List[any], band: int = 1, relative_to_root_path: bool = True) -> Tuple[rasterio.DatasetReader, np.ndarray]:        
        path = self.get_file_path(filename, relative_to_root_path)
        if self.verbose_reporting:
            print(f"{Fore.WHITE}{Style.DIM}READING {path}{Style.RESET_ALL}")
                
        raster_reader = rasterio.open(path)
        band_data = raster_reader.read(band)

        for nodata_value in nodata_values:
            fill_value = self.nodata_value
            if nodata_value != self.nodata_value:
                print(f"{Fore.YELLOW}{Style.DIM}REPLACING NODATA VALUE {nodata_value} WITH FILL VALUE {fill_value}{Style.RESET_ALL}")                     
                band_data = np.where(band_data==nodata_value, fill_value, band_data)

        return raster_reader, band_data

    def _write_file(self, filename, outdata, out_metadata, relative_to_root_path: bool = True):
        path = self.get_file_path(filename, relative_to_root_path)
        RecreatBase.write_output(path, outdata, out_metadata)

    def _get_matrix(self, fill_value: float, shape: Tuple[int,int], dtype: any) -> np.ndarray:
        out_rst = np.full(shape=shape, fill_value=fill_value, dtype=dtype)
        return out_rst       

    def _get_shape(self) -> Tuple[int,int]:
        return self.land_use_map_reader.shape
    
    def _get_metadata(self, new_dtype, new_nodata_value):
        out_meta = self.land_use_map_reader.meta.copy()
        out_meta.update({
            'nodata' : new_nodata_value,
            'dtype' : new_dtype
        })
        return out_meta        

    def _get_data_object(self, filename, nodata_replacement_value: any = None) -> np.ndarray:
        data_rader, data_mtx = self._get_file(filename, [self.nodata_value])
        if nodata_replacement_value is not None:
            data_mtx[data_mtx == self.nodata_value] = nodata_replacement_value
        return data_mtx

    def _get_supply_for_land_use_class_and_cost(self, lu: int, cost: int, return_cost_window_difference: bool = False) -> np.ndarray:        
        lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"        
        
        filename: str = None
        if not return_cost_window_difference:
            filename = f"SUPPLY/totalsupply_class_{lu}_cost_{cost}_clumped.tif" if lu_type == 'patch' else f"SUPPLY/totalsupply_edge_class_{lu}_cost_{cost}_clumped.tif"
        else:
            filename = f"SUPPLY/totalsupply_{lu}_within_cost_range_{cost}.tif"
        
        return self._get_data_object(filename, nodata_replacement_value=0)
    
    def _get_lu_patches(self, lu: int) -> np.ndarray:
        filename = f"CLUMPS_LU/clumps_{lu}.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)

    def _get_disaggregated_population(self) -> np.ndarray:
        filename = f"DEMAND/disaggregated_population.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)

    def _get_supply_for_cost(self, cost: int, return_cost_window_difference: bool = False) -> np.ndarray:
        filename = f"SUPPLY/totalsupply_cost_{cost}.tif" if not return_cost_window_difference else f"SUPPLY/totalsupply_within_cost_range_{cost}.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)

    def _get_lu_weighted_supply_for_cost(self, cost: int, return_cost_window_difference: bool = False) -> np.ndarray:
        filename = f"SUPPLY/lu_weighted_totalsupply_cost_{cost}.tif" if not return_cost_window_difference else f"SUPPLY/lu_weighted_totalsupply_within_cost_range_{cost}.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)

    def _get_land_use_class_mask(self, lu: int) -> np.ndarray:        
        lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
        filename = f"MASKS/mask_{lu}.tif" if lu_type == 'patch' else f"MASKS/edges_{lu}.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)
    
    def _get_diversity_for_cost(self, cost: int, return_cost_window_difference: bool = False) -> np.ndarray:
        filename = f"DIVERSITY/diversity_cost_{cost}.tif" if not return_cost_window_difference else f"DIVERSITY/diversity_within_cost_range_{cost}.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)
    
    def _get_beneficiaries_for_cost(self, cost: int, return_cost_window_difference: bool = False) -> np.ndarray:
        filename = f"DEMAND/beneficiaries_within_cost_{cost}.tif" if not return_cost_window_difference else f"DEMAND/beneficiaries_within_cost_range_{cost}.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)

    def _get_proximity_raster_for_lu(self, lu) -> np.ndarray:
        filename = f"PROXIMITY/dr_{lu}.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)
    
    def _get_minimum_cost_for_lu(self, lu) -> np.ndarray:
        filename = f'COSTS/minimum_cost_{lu}.tif'
        return self._get_data_object(filename)
    
    def _get_flow_for_land_use_class_and_cost(self, lu, cost) -> np.ndarray:
        lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
        filename = "FLOWS/flow_class_{}_cost_{}.tif".format(lu, cost) if lu_type == 'patch' else "FLOWS/flow_edge_class_{}_cost_{}.tif".format(lu, cost)
        return self._get_data_object(filename, nodata_replacement_value=0)

    def _get_flow_for_cost(self, cost: int, return_cost_window_difference: bool = False) -> np.ndarray:
        filename = f"FLOWS/flow_for_cost_{cost}.tif" if not return_cost_window_difference else f"FLOWS/flow_within_cost_range_{cost}.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)

    def _get_clumps(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.clumps_matrix is None:            
            if not os.path.isfile(self.get_file_path('BASE/clumps.tif')):
                raise FileNotFoundError('Clumps not found in BASE folder. Run clump detection to create this dataset.')
                        
            clumps_reader, self.clumps_matrix = self._get_file('BASE/clumps.tif', [self.nodata_value])
            self.clumps_nodata_mask = np.isin(self.clumps_matrix, [self.nodata_value], invert=False)
        
        return self.clumps_matrix, self.clumps_nodata_mask

    def _get_land_use(self) -> np.ndarray:
        if self.land_use_map_matrix is None:
            if not os.path.isfile(self.get_file_path('BASE/lulc.tif')):
                raise FileNotFoundError('Land use/land cover map not found in BASE folder. Run land use map alignment to create this dataset.')
            
            land_use_reader, self.land_use_map_matrix = self._get_file("BASE/lulc.tif", [self.nodata_value]) 
        
        return self.land_use_map_matrix

#endregion

#region land_use_map handling

    def set_land_use_map(self, root_path: str, land_use_filename: str) -> None:
        """Set root path and land-use map to be used in the analysis. 

        :param root_path: Path to the root folder containing the land-use map file, and into which results will be written.
        :type root_path: str
        :param land_use_filename: Filename of the land-use map raster file, that should be placed into the root path.
        :type land_use_filename: str
        """
        # set root path
        self.root_path = root_path                
        # make folders in root path
        self.make_environment()                 
        
        # get reference to the land use/land cover file
        lulc_file_path = self.get_file_path(land_use_filename, relative_to_root_path=True)
        self.land_use_map_reader = rasterio.open(lulc_file_path)

    def align_land_use_map(self, nodata_values: List[int], band: int = 1, reclassification_mappings: Dict[int, List[int]] = None):
        """Pre-process land-use map and conduct a reclassification of land-use class values, if needed. Nodata values will be reclassified to internal nodata value (by default: -9999, or as set using set_params method). 
           The pre-processed land-use map will be written into the BASE folder as lulc.tif.

        :param nodata_values: List of values to be treated as nodata.
        :type nodata_values: List[int]
        :param band: Raster band holding land-use class values, defaults to 1.
        :type band: int, optional
        :param reclassification_mappings: Dictionary mapping new class values to a list of one or more class values to be reclassified, defaults to None.
        :type reclassification_mappings: Dict[int, List[int]], optional
        """

        self.printStepInfo("Aligning land-use map")
        
        # conduct this if there is no base lulc file
        lulc_file_path = self.get_file_path("BASE/lulc.tif")
        if not os.path.isfile(lulc_file_path):
         
            # get lulc data from datasetreader 
            lulc_data = self.land_use_map_reader.read(band)

            # conduct value replacement
            for nodata_value in nodata_values:
                fill_value = self.nodata_value
                if nodata_value != self.nodata_value:
                    print(f"{Fore.YELLOW}{Style.DIM}REPLACING NODATA VALUE {nodata_value} WITH FILL VALUE {fill_value}{Style.RESET_ALL}")                     
                    lulc_data = np.where(lulc_data==nodata_value, fill_value, lulc_data)

            # conduct recategorization of values, if requested
            if reclassification_mappings is not None:
                lulc_data = self.reclassify(lulc_data, reclassification_mappings)

            # write out result to be re-used later
            self._write_file("BASE/lulc.tif", lulc_data.astype(np.int32), self._get_metadata(np.int32, self.nodata_value))

        self.printStepCompleteInfo()

    def reclassify(self, data_mtx: np.ndarray, mappings: Dict[int, List[int]]) -> np.ndarray:
        """Reclassify land-use values in land-use map.

        :param data_mtx: Array of land-use class values.
        :type data_mtx: np.ndarray
        :param mappings: Dictionary mapping new class values to a list of one or more class values to be reclassified. 
        :type mappings: Dict[int, List[int]]
        :return: Array of land-use class values, reclassified according to mappings.
        :rtype: np.ndarray
        """
        self.printStepInfo("Raster reclassification")        
        current_task = self._new_task("[white]Reclassification", total=len(mappings.keys()))

        # iterate over key-value combinations
        with self.progress:            
            for (new_class_value, classes_to_aggregate) in mappings.items():
                replacement_mask = np.isin(data_mtx, classes_to_aggregate, invert=False)
                data_mtx[replacement_mask] = new_class_value
                del replacement_mask                
                self.progress.update(current_task, advance=1)                 

        # done
        self.taskProgressReportStepCompleted()
        return data_mtx

#endregion

#region basic processing

    def detect_clumps(self, barrier_classes: List[int], contiguity: Contiguity = Contiguity.Queen) -> None:
        """Detect clumps as contiguous areas in the land-use raster that are separated by the specified barrier land-uses. Connectivity is defined as queens contiguity. 

        :param barrier_classes: Classes acting as barriers, i.e., separating clumps, defaults to [0]
        :type barrier_classes: List[int], optional
        """    
        self.printStepInfo("Detecting clumps")

        lulc_data = self._get_land_use()
        nr_clumps, out_clumps = self._detect_clumps_in_raster(lulc_data, barrier_classes=barrier_classes, contiguity=contiguity)        
        self._write_file("BASE/clumps.tif", out_clumps, self._get_metadata(np.int32, self.nodata_value))        
        
        # done
        self.taskProgressReportStepCompleted()

    def _detect_clumps_in_raster(self, mtx_data: np.ndarray, barrier_classes: List[int], contiguity: Contiguity) -> Tuple[int, np.ndarray]:
        
        # barrier_classes are user-defined classes as well as nodata parts
        # mask raster accordingly
        barrier_classes += [self.nodata_value]
        barriers_mask = np.isin(mtx_data, barrier_classes, invert=False)
        mtx_data[barriers_mask] = 0

        # determine patches
        # by default, setup kernel for queen contiguity
        clump_connectivity = np.full((3,3), 1)
        # modify kernel for rook contiguity
        if contiguity is Contiguity.Rook:
            clump_connectivity[0][0] = 0
            clump_connectivity[0][2] = 0
            clump_connectivity[2][0] = 0
            clump_connectivity[2][2] = 0

        out_clumps = self._get_matrix(fill_value=0, shape=self._get_shape(), dtype=np.int32)
        nr_clumps = ndimage.label(mtx_data, structure=clump_connectivity, output=out_clumps)

        # update clumps to hold nodata value where clump=0 (= background)
        out_clumps[out_clumps == 0] = self.nodata_value

        print(f"{Fore.YELLOW}{Style.BRIGHT} {nr_clumps} CLUMPS FOUND{Style.RESET_ALL}")
        return nr_clumps, out_clumps




    def mask_landuses(self, lu_classes: List[int] = None) -> None:
        """Generate land-use class masks (occurrence masks) for patch, edge, and built-up land-use classes.
        
        :param lu_classes: Classes for which to create class masks, by default None. If None, create class masks for all patch classes and edge classes.
        :type lu_classes: List[int], optional        
        """

        classes_for_masking = self.lu_classes_recreation_edge + self.lu_classes_recreation_patch if lu_classes is None else lu_classes

        # mask classes of interest into a binary raster to indicate presence/absence of recreational potential
        # we require this for all classes relevant to processing: patch and edge recreational classes, built-up classes
        self.printStepInfo("Creating land-use class masks")
        current_task = self._new_task('[white]Masking', total=len(classes_for_masking))

        with self.progress:
            
            # import land-use dataset
            lulc_data = self._get_land_use()
            # import clump dataset for masking of outputs           
            clump_data, clump_nodata_mask = self._get_clumps()

            for lu in classes_for_masking:
                
                current_lu_mask = self._get_matrix(0, self._get_shape(), np.int32)

                # make mask for relevant pixels
                mask = np.isin(lulc_data, [lu], invert=False)                
                # mask with binary values 
                current_lu_mask[mask] = 1
                # mask with clump nodata
                current_lu_mask[clump_nodata_mask] = self.nodata_value
                
                # write to disk
                self._write_file(f"MASKS/mask_{lu}.tif", current_lu_mask, self._get_metadata(np.int32, self.nodata_value))

                self.progress.update(current_task, advance=1)

        # done    
        self.taskProgressReportStepCompleted()
    
    def detect_edges(self, lu_classes: List[int] = None, ignore_edges_to_class: int = None, buffer_edges: List[int] = None) -> None:
        """Detect edges (patch perimeters) of land-use classes that are defined as edge classes. For classes contained in the buffer_edges list, buffer patch perimeters by one pixel.

        :param lu_classes: List of classes for which edges should be assessed, by default None. If None, all classes specified as classes.edge will be processed.
        :type lu_classes: List[int], optional
        :param ignore_edges_to_class: Class to which edges should be ignored, defaults to None.
        :type ignore_edges_to_class: int, optional
        :param buffer_edges: List of classes for which edges should be buffered (expanded by one pixel), defaults to None. Mostly, classes considered as barriers in the clump detection that also represent recreational opportunities should be considered here.
        :type buffer_edges: List[int], optional
        """        

        # determine edge pixels of edge-only classes such as water opportunities

        classes_to_assess = lu_classes if lu_classes is not None else self.lu_classes_recreation_edge
        
        if (len(classes_to_assess) > 0):
            
            if ignore_edges_to_class is None:
                self.clib.div_filter.restype = ctypes.c_int
                self.clib.div_filter.argtypes = (
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_void_p,
                )
            else:
                self.clib.div_filter_ignore_class.restype = ctypes.c_int
                self.clib.div_filter_ignore_class.argtypes = (
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_void_p,
                )

            self.printStepInfo("Detecting edges")
            current_task = self._new_task("[white]Detecting edges", total=len(classes_to_assess))

            with self.progress:
                
                # import lulc 
                lulc_data = self._get_land_use()                
                # import clump dataset for masking of outputs
                clump_data, clump_nodata_mask = self._get_clumps()
                
                for lu in classes_to_assess:
                    
                    if ignore_edges_to_class is None:                    
                        user_data = ctypes.c_double(lu)
                        ptr = ctypes.cast(ctypes.pointer(user_data), ctypes.c_void_p)
                        div_filter = LowLevelCallable(self.clib.div_filter, user_data=ptr, signature="int (double *, intptr_t, double *, void *)")              
                    else:
                        user_values = [lu, 0]
                        user_data = (ctypes.c_int * 10)(*user_values)
                        ptr = ctypes.cast(ctypes.pointer(user_data), ctypes.c_void_p)
                        div_filter = LowLevelCallable(self.clib.div_filter_ignore_class, user_data=ptr, signature="int (double *, intptr_t, double *, void *)")              

                    # apply a 3x3 rectangular sliding window to determine pixel value diversity in window
                    rst_edgePixelDiversity = self._moving_window_generic(data_mtx=lulc_data, kernel_func=div_filter, kernel_size=3, kernel_shape='rect', dest_datatype=np.int32)
                    rst_edgePixelDiversity = rst_edgePixelDiversity - 1
                    rst_edgePixelDiversity[rst_edgePixelDiversity > 1] = 1                

                    # depending on whether to grow edge or not, intersect with land-use mask to have edge within land-use, or
                    # extending outside.
                    if lu in buffer_edges:
                        rst_edgePixelDiversity[clump_nodata_mask] = self.nodata_value
                        self._write_file(f"MASKS/edges_{lu}.tif", rst_edgePixelDiversity.astype(np.int32), self._get_metadata(np.int32, self.nodata_value))                        
                    
                    else:
                        # read masking raster, reconstruct original data by replacing nodata values with 0
                        mask_reader, mask_data = self._get_file(f"MASKS/mask_{lu}.tif", [self.nodata_value]) 
                        mask_data[clump_nodata_mask] = 0
                        mask_data = mask_data * rst_edgePixelDiversity
                        mask_data[clump_nodata_mask] = self.nodata_value

                        self._write_file(f"MASKS/edges_{lu}.tif", mask_data.astype(np.int32), self._get_metadata(np.int32, self.nodata_value))
                        del mask_data

                    # some cleaning
                    del rst_edgePixelDiversity
                    self.progress.update(current_task, advance=1)

            # done
            self.taskProgressReportStepCompleted()

#endregion

#region Moving-window operations

    def _get_circular_kernel(self, kernel_size: int) -> np.ndarray:
        
        kernel = np.zeros((kernel_size,kernel_size))
        radius = kernel_size/2
        # modern scikit uses a tuple for center
        rr, cc = disk( (kernel_size//2, kernel_size//2), radius)
        kernel[rr,cc] = 1
        return kernel
    
    def _moving_window_generic(self, data_mtx: np.ndarray, kernel_func: Callable[[np.ndarray], float], kernel_size: int, kernel_shape: str = 'circular', dest_datatype = None) -> np.ndarray:
        
        dtype_to_use = np.float32
        # make kernel
        kernel = self._get_circular_kernel(kernel_size) if kernel_shape == 'circular' else np.full((kernel_size, kernel_size), 1)
        mtx_res = self._get_matrix(0, data_mtx.shape, dtype_to_use)        
        # apply moving window over input mtx
        ndimage.generic_filter(data_mtx.astype(dtype_to_use), kernel_func, footprint=kernel, output=mtx_res, mode='constant', cval = 0)
        return mtx_res
    
    def _moving_window_convolution(self, data_mtx: np.ndarray, kernel_size: int, kernel_shape: str = 'circular') -> np.ndarray: 
        
        dtype_to_use = np.float32
        # make kernel
        kernel = self._get_circular_kernel(kernel_size) if kernel_shape == 'circular' else np.full((kernel_size, kernel_size), 1)
        # make result matrix
        mtx_res = self._get_matrix(0, data_mtx.shape, dtype_to_use)
        # apply convolution filter from ndimage that sums as weights are 0 or 1.        
        ndimage.convolve(data_mtx.astype(dtype_to_use), kernel, output=mtx_res, mode = 'constant', cval = 0)        
        return mtx_res
    
    def _moving_window_filter2d(self, data_mtx: np.ndarray, kernel_size: int, kernel_shape: str = 'circular') -> np.ndarray: 
        
        # make kernel
        radius = int(kernel_size / 2)
        kernel = self._get_circular_kernel(kernel_size) if kernel_shape == 'circular' else np.full((kernel_size, kernel_size), 1)
        # make sure that input is padded, as this determines border values
        data_mtx = np.pad(data_mtx, radius, mode='constant')
        
        if data_mtx.dtype == np.int32 or data_mtx.dtype == np.int64:
            data_mtx = data_mtx.astype(np.float32)

        mtx_res = cv.filter2D(data_mtx, -1, cv.flip(kernel, -1)) 
        return mtx_res[radius:-radius,radius:-radius]
    
    def _sum_values_in_kernel(self, mtx_source: np.ndarray, mtx_clumps: np.ndarray, clump_slices: List[any], cost: float, mode: str, progress_task: any = None) -> np.ndarray:
        
        # grid to store summed values in kernel 
        mtx_result = self._get_matrix(0, self._get_shape(), mtx_source.dtype)               

        # use lowlevelcallable to speed up moving window operation               
        if mode == 'generic_filter':
            self.clib.sum_filter.restype = ctypes.c_int
            self.clib.sum_filter.argtypes = (
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_void_p,
            )
            sum_filter = LowLevelCallable(self.clib.sum_filter, signature="int (double *, intptr_t, double *, void *)") 

        # now operate over clumps, in order to safe some computational time
        for patch_idx in range(len(clump_slices)):
            
            obj_slice = clump_slices[patch_idx]
            obj_label = patch_idx + 1

            # get slice from land-use mask
            sliced_mtx_source = mtx_source[obj_slice].copy() 
            sliced_mtx_clumps = mtx_clumps[obj_slice]

            # properly mask out current object
            obj_mask = np.isin(sliced_mtx_clumps, [obj_label], invert=False)
            sliced_mtx_source[~obj_mask] = 0

            # now all pixels outside of clump should be zeroed, and we can determine total supply within sliding window            
            if mode == 'convolve':            
                sliding_results = self._moving_window_convolution(sliced_mtx_source, cost)
            elif mode == 'generic_filter':                 
                sliding_results = self._moving_window_generic(sliced_mtx_source, sum_filter, cost)
            elif mode == 'ocv_filter2d':
                sliding_results = self._moving_window_filter2d(sliced_mtx_source, cost)
           
            # similarly, mask all pixels outside of clump to 0 to assert result constraint to current clump 
            sliding_results[~obj_mask] = 0
            mtx_result[obj_slice] += sliding_results.astype(mtx_source.dtype)
            
            del sliding_results
            del sliced_mtx_source

            if progress_task is not None:
                self.progress.update(progress_task, advance=1)
        
        # done with current iterations. return result
        del mtx_source
        return mtx_result
    
#endregion

#region supply-related methods

    def class_total_supply(self, lu_classes: List[int] = None, mode: str = 'ocv_filter2d') -> None:
        """Determine class total supply.

        :param lu_classes: List of classes for which supply should be determined, by default None. If None, determine supply for all patch and edge classes.
        :type lu_classes: List[int], optional
        :param mode: Method to perform moving window operation. One of 'generic_filter', 'convolve', or 'ocv_filter2d', defaults to 'ocv_filter2d'.
        :type mode: str, optional
        """        

        dtype_to_use = np.int32

        # for each recreation patch class and edge class, determine total supply within cost windows
        # do this for each clump, i.e., operate only on parts of masks corresponding to clumps, ignore patches/edges external to each clump
        self.printStepInfo("Determining clumped supply per class")
        
        # clumps are required to properly mask islands
        rst_clumps, clump_nodata_mask = self._get_clumps()
        clump_slices = ndimage.find_objects(rst_clumps.astype(np.int64))
        
        lu_classes = lu_classes if lu_classes is not None else (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge)

        step_count = len(clump_slices) * len(lu_classes) * len(self.cost_thresholds)
        current_task = self._new_task("[white]Determining clumped supply", total=step_count)

        with self.progress:
            for c in self.cost_thresholds: 
                for lu in lu_classes:    
                    
                    # determine source of list
                    lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
                    lu_mask = self._get_land_use_class_mask(lu)

                    # get result of windowed operation
                    lu_supply_mtx = self._sum_values_in_kernel(
                        mtx_source=lu_mask.astype(dtype_to_use), 
                        mtx_clumps=rst_clumps, 
                        clump_slices=clump_slices, 
                        cost=c, 
                        mode=mode, 
                        progress_task=current_task
                    )

                    # mask nodata regions based on clumps
                    lu_supply_mtx[clump_nodata_mask] = self.nodata_value
                   
                    # export current cost
                    outfile_name = (
                        f"SUPPLY/totalsupply_class_{lu}_cost_{c}_clumped.tif"
                        if lu_type == "patch"
                        else f"SUPPLY/totalsupply_edge_class_{lu}_cost_{c}_clumped.tif"
                    )
                    self._write_file(outfile_name, lu_supply_mtx.astype(dtype_to_use), self._get_metadata(dtype_to_use, self.nodata_value))
                    del lu_supply_mtx           


        # done
        self.taskProgressReportStepCompleted()

    def _get_aggregate_class_total_supply_for_cost(self, cost, lu_weights = None, write_non_weighted_result = True, task_progress = None):                        
        
        current_total_supply_at_cost = None
        current_weighted_total_supply_at_cost = None

        # make grids for the results: zero-valued grids with full lsm extent
        if write_non_weighted_result:
            current_total_supply_at_cost = self._get_matrix(0, self._get_shape(), np.float64) 
        if lu_weights is not None:
            current_weighted_total_supply_at_cost = self._get_matrix(0, self._get_shape(), np.float64)
        
        for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):                
            
            # determine source of list
            lu_supply_mtx = self._get_supply_for_land_use_class_and_cost(lu, cost)
            lu_supply_mtx = lu_supply_mtx.astype(np.float64)

            # add to aggregations
            if write_non_weighted_result:
                current_total_supply_at_cost += lu_supply_mtx
            if lu_weights is not None:
                current_weighted_total_supply_at_cost += (lu_supply_mtx * lu_weights[lu])

            if task_progress is not None:
                self.progress.update(task_progress, advance=1)

        if lu_weights is not None:
            current_weighted_total_supply_at_cost = current_weighted_total_supply_at_cost / sum(lu_weights.values())

        # return aggregated grids for given cost
        return current_total_supply_at_cost, current_weighted_total_supply_at_cost
    
    def aggregate_class_total_supply(self, lu_weights: Dict[any,float] = None, write_non_weighted_result: bool = True) -> None:
        """Aggregate total supply of land-use classes within each specified cost threshold. A weighting schema may be supplied, in which case a weighted average is determined as the sum of weighted class supply divided by the sum of all weights.

        :param lu_weights: Dictionary of land-use class weights, where keys refer to land-use classes, and values to weights. If specified, weighted total supply will be determined, defaults to None.
        :type lu_weights: Dict[any,float], optional
        :param write_non_weighted_result: Indicates if non-weighted total supply be computed, defaults to True.
        :type write_non_weighted_result: bool, optional
        """        
                
        self.printStepInfo('Determining clumped total supply')

        # progress reporting        
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge))
        current_task = self._new_task("[white]Aggregating clumped supply", total=step_count)

        with self.progress:
            
            mtx_clumps, clump_nodata_mask = self._get_clumps()
                        
            for c in self.cost_thresholds:
                # get aggregation for current cost threshold
                current_total_supply_at_cost, current_lu_weighted_total_supply_at_cost = self._get_aggregate_class_total_supply_for_cost(cost=c, lu_weights=lu_weights, write_non_weighted_result=write_non_weighted_result, task_progress=current_task)                                           

                # mask nodata areas and export total for costs
                if write_non_weighted_result:  
                    current_total_supply_at_cost[clump_nodata_mask] = self.nodata_value  
                    self._write_file(f"SUPPLY/totalsupply_cost_{c}.tif", current_total_supply_at_cost, self._get_metadata(np.float64, self.nodata_value))
                
                # export weighted total, if applicable
                if lu_weights is not None:                    
                    current_lu_weighted_total_supply_at_cost[clump_nodata_mask] = self.nodata_value
                    self._write_file(f"SUPPLY/lu_weighted_totalsupply_cost_{c}.tif", current_lu_weighted_total_supply_at_cost, self._get_metadata(np.float64, self.nodata_value))


        # additionally, compute differences in supply across cost ranges, if needed
        if write_non_weighted_result:
            current_template = Template("SUPPLY/totalsupply_within_cost_range_${cost}.tif")
            self._get_cost_range_differences(self._get_supply_for_cost, current_template)

        # additionally, compute differences in lu-weighted supply across cost ranges, if needed
        if lu_weights is not None:
            current_template = Template("SUPPLY/lu_weighted_totalsupply_within_cost_range_${cost}.tif")
            self._get_cost_range_differences(self._get_lu_weighted_supply_for_cost, current_template)

        # done
        self.taskProgressReportStepCompleted()
    
    def average_total_supply_across_cost(self, land_use_weighted_supply_as_input: bool = False, cost_weights: Dict[float, float] = None, write_non_weighted_result: bool = True, write_scaled_result: bool = True) -> None:
        """Determine the total supply of recreational opportunities, averaged across cost thresholds. The average may consider either land-use type-weighted aggregated supply, or raw (non-weighted) aggregated supply, as determined through the aggregate_class_total_supply method.
           A cost-weighting of supply may be conducted. 

        :param land_use_weighted_supply_as_input: Indicates whether land-use weighted aggregated supply should be used (if set to True), or raw (non-weighted) aggregated supply is used (if set to False), by default False.
        :type land_use_weighted_supply_as_input: bool, optional
        :param cost_weights: Dictionary of cost weights, where keys refer to cost thresholds, and values to weights, by default None.
        :type cost_weights: Dict[float, float], optional
        :param write_non_weighted_result: Indicates if non-weighted total supply should be computed, defaults to True.
        :type write_non_weighted_result: bool, optional
        :param write_scaled_result: Indicates if min-max-scaled values should be written as separate outputs, defaults to True
        :type write_scaled_result: bool, optional
        """        

        # this averaging method will create the following datasets, depending on specified parameters
        # if cost_range_differences_as_input is False:
        # ... non-weighted average of supply in cost
        # ... cost-weighted average of supply in cost if cost weights is not None
        # if cost_range_differences_as_input is True:
        # ... non-weighted average of supply within cost ranges 
        # ... cost-weighted average of supply within cost ranges
        # either of the above products use...
        # ... either non-lu-weighted inputs if land_use_weighted_supply_as_input is False
        # ... or     lu-weighted supply if land_use_weighted_supply_as_input is True


        self.printStepInfo("Averaging supply across costs")
        step_count = len(self.cost_thresholds)
        current_task = self._new_task("[white]Averaging supply", total=step_count)

        with self.progress as p:

            # clumps are required to properly mask islands
            rst_clumps, clump_nodata_mask = self._get_clumps()

            # prepare result rasters
            if write_non_weighted_result:
                non_weighted_average_total_supply = self._get_matrix(0, self._get_shape(), np.float64)           
            if cost_weights is not None:
                cost_weighted_average_total_supply = self._get_matrix(0, self._get_shape(), np.float64)               

            # iterate over costs
            for c in self.cost_thresholds:

                # re-aggregate lu supply within cost, using currently supplied weights
                # mtx_current_cost_total_supply, mtx_current_cost_weighted_total_supply = self._get_aggregate_class_total_supply_for_cost(c, lu_weights, write_non_weighted_result, current_task)                                           
                if write_non_weighted_result:
                    mtx_current_cost_supply = self._get_lu_weighted_supply_for_cost(c, return_cost_window_difference=False) if land_use_weighted_supply_as_input else self._get_supply_for_cost(c, return_cost_window_difference=False)
                    non_weighted_average_total_supply += mtx_current_cost_supply
                
                if cost_weights is not None:
                    mtx_current_cost_supply = self._get_lu_weighted_supply_for_cost(c, return_cost_window_difference=True) if land_use_weighted_supply_as_input else self._get_supply_for_cost(c, return_cost_window_difference=True)
                    cost_weighted_average_total_supply += (mtx_current_cost_supply * cost_weights[c])
                  
            # complete determining averages for the various combinations
            # def. case
            if write_non_weighted_result:
                non_weighted_average_total_supply = non_weighted_average_total_supply / len(self.cost_thresholds)
                non_weighted_average_total_supply[clump_nodata_mask] = self.nodata_value
                self._write_file("INDICATORS/non_weighted_avg_totalsupply.tif", non_weighted_average_total_supply, self._get_metadata(np.float64, self.nodata_value))
                
                # if write_scaled_result:
                #     # apply min-max scaling
                #     scaler = MinMaxScaler()
                #     non_weighted_average_total_supply = scaler.fit_transform(non_weighted_average_total_supply.reshape([-1,1]))
                #     self._write_dataset('INDICATORS/scaled_non_weighted_avg_totalsupply.tif', non_weighted_average_total_supply.reshape(self.lsm_mtx.shape))


            # def. case + cost weighting
            if cost_weights is not None:
                cost_weighted_average_total_supply = cost_weighted_average_total_supply / sum(cost_weights.values())
                cost_weighted_average_total_supply[clump_nodata_mask] = self.nodata_value
                self._write_file("INDICATORS/cost_weighted_avg_totalsupply.tif", cost_weighted_average_total_supply, self._get_metadata(np.float64, self.nodata_value))
                
                # if write_scaled_result:
                #     # apply min-max scaling
                #     scaler = MinMaxScaler()
                #     cost_weighted_average_total_supply = scaler.fit_transform(cost_weighted_average_total_supply.reshape([-1,1]))
                #     self._write_dataset('INDICATORS/scaled_cost_weighted_avg_totalsupply.tif', cost_weighted_average_total_supply.reshape(self.lsm_mtx.shape))
            
        # done
        self.taskProgressReportStepCompleted()

#endregion

#region diversity

    def class_diversity(self) -> None:
        """Determine the diversity of land-use classes with assumed recreational potential within each cost threshold. 
           Additionally, differences between cost thresholds are computed.
        """

        self.printStepInfo("Determining class diversity within costs")        

        step_count = (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch)) * len(self.cost_thresholds)
        current_task = self._new_task("[white]Determining class diversity", total=step_count)

        with self.progress as p:

            # clumps are required to properly mask islands
            rst_clumps, clump_nodata_mask = self._get_clumps()

            for c in self.cost_thresholds:    
                
                mtx_diversity_at_cost = self._get_matrix(0, self._get_shape(), np.int32)

                for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):                
                    
                    # determine opportunity type and get respective dataset
                    mtx_supply = self._get_supply_for_land_use_class_and_cost(lu, c)                   
                    
                    mtx_supply[mtx_supply > 0] = 1
                    mtx_diversity_at_cost += mtx_supply.astype(np.int32)
                    
                    p.update(current_task, advance=1)

                # mask using clump nodata mask
                mtx_diversity_at_cost[clump_nodata_mask] = self.nodata_value

                # export current cost diversity                                
                self._write_file(f"DIVERSITY/diversity_cost_{c}.tif", mtx_diversity_at_cost, self._get_metadata(np.int32, self.nodata_value))
                del mtx_diversity_at_cost

        # additionally, compute differences in diversity across cost ranges
        current_template = Template("DIVERSITY/diversity_within_cost_range_${cost}.tif")
        self._get_cost_range_differences(self._get_diversity_for_cost, current_template)

        # done
        self.taskProgressReportStepCompleted()
    
    def average_diversity_across_cost(self, cost_weights: Dict[float, float] = None, write_non_weighted_result: bool = True, write_scaled_result: bool = True) -> None:
        """Determine diversity of (recreational) land-uses averaged across cost thresholds. 

        :param cost_weights: Dictionary of cost weights, where keys refer to cost thresholds, and values to weights, by default None.
        :type cost_weights: Dict[float, float], optional
        :param write_non_weighted_result: Indicates if non-weighted total supply should be computed, defaults to True.
        :type write_non_weighted_result: bool, optional
        :param write_scaled_result: Indicates if min-max-scaled values should be written as separate outputs, defaults to True.
        :type write_scaled_result: bool, optional
        """        

        self.printStepInfo("Averaging diversity across costs")
        step_count = len(self.cost_thresholds)
        current_task = self._new_task("[white]Averaging diversity", total=step_count)

        with self.progress as p:

            # clumps are required to properly mask islands
            rst_clumps, clump_nodata_mask = self._get_clumps()

            # result raster
            if write_non_weighted_result:
                average_diversity = self._get_matrix(0, self._get_shape(), np.float64)
            if cost_weights is not None:
                cost_weighted_average_diversity = self._get_matrix(0, self._get_shape(), np.float64)

            # iterate over cost thresholds and aggregate cost-specific diversities into result
            for c in self.cost_thresholds:
                
                if write_non_weighted_result:
                    mtx_current_diversity = self._get_diversity_for_cost(c, return_cost_window_difference=False) 
                    average_diversity += mtx_current_diversity.astype(np.float64)
                
                if cost_weights is not None:
                    mtx_current_diversity = self._get_diversity_for_cost(c, return_cost_window_difference=True) 
                    cost_weighted_average_diversity += (average_diversity.astype(np.float64) * cost_weights[c])
                
                p.update(current_task, advance=1)


            # export averaged diversity grids
            if write_non_weighted_result:
                average_diversity = average_diversity / len(self.cost_thresholds)
                average_diversity[clump_nodata_mask] = self.nodata_value
                self._write_file("INDICATORS/non_weighted_avg_diversity.tif", average_diversity, self._get_metadata(np.float64, self.nodata_value))
                
                # if write_scaled_result:
                #     # apply min-max scaling
                #     scaler = MinMaxScaler()
                #     average_diversity = scaler.fit_transform(average_diversity.reshape([-1,1]))
                #     self._write_dataset('INDICATORS/scaled_non_weighted_avg_diversity.tif', average_diversity.reshape(self.lsm_mtx.shape))
                        
            if cost_weights is not None:
                cost_weighted_average_diversity = cost_weighted_average_diversity / sum(cost_weights.values())
                cost_weighted_average_diversity[clump_nodata_mask] = self.nodata_value
                self._write_file("INDICATORS/cost_weighted_avg_diversity.tif", cost_weighted_average_diversity, self._get_metadata(np.float64, self.nodata_value))
                
                # if write_scaled_result:
                #     # apply min-max scaling
                #     scaler = MinMaxScaler()
                #     cost_weighted_average_diversity = scaler.fit_transform(cost_weighted_average_diversity.reshape([-1,1]))
                #     self._write_dataset('INDICATORS/scaled_cost_weighted_avg_diversity.tif', cost_weighted_average_diversity.reshape(self.lsm_mtx.shape))

        # done
        self.taskProgressReportStepCompleted()

#endregion

#region disaggregation and demand

    def disaggregation(self, population_grid: str, disaggregation_method: DisaggregationMethod, max_pixel_count: int, write_scaled_result: bool = True, count_threshold: int = None, min_sample_size: int = None) -> None:
        """Disaggregates population to specified built-up (residential) classes. 

        :param population_grid: Name of the population raster file to be used for disaggregation. Each cell in this raster is considered a source zone.
        :type population_grid: str
        :param disaggregation_method: Method to conduct disaggregation.
        :type disaggregation_method: DisaggregationMethod
        :param max_pixel_count: Number of built-up pixels per population raster. 
        :type max_pixel_count: int
        :param write_scaled_result: Indicates if min-max-scaled values should be written as separate outputs, defaults to True.
        :type write_scaled_result: bool, optional
        :param count_threshold: Sampling threshold, i.e., number of built-up pixels within a source zone required for inclusion in the class sampling for relative class density computation, by default None. Required for intelligent dasymetric mapping.
        :type count_threshold: int, optional
        :param min_sample_size: Minimum number of samples for the computation of relative class density, by default None. Required for intelligent dasymetric mapping. 
        :type min_sample_size: int, optional
        """
        # mask residential classes
        self.mask_landuses(lu_classes=self.lu_classes_builtup)

        if disaggregation_method is DisaggregationMethod.SimpleAreaWeighted:            
            disaggregation_engine = SimpleAreaWeightedEngine(
                data_path=self.data_path, 
                root_path=self.root_path,
                population_grid=population_grid, 
                residential_classes=self.lu_classes_builtup, 
                max_pixel_count=max_pixel_count,
                nodata_value=self.nodata_value,
                write_scaled_result=write_scaled_result
            )
            
            disaggregation_engine.run()

        elif disaggregation_method is DisaggregationMethod.IntelligentDasymetricMapping:
            disaggregation_engine = DasymetricMappingEngine(
                data_path=self.data_path, 
                root_path=self.root_path, 
                population_grid=population_grid,
                residential_classes=self.lu_classes_builtup, 
                max_pixel_count=max_pixel_count,
                count_threshold=count_threshold,
                min_sample_size=min_sample_size,
                nodata_value=self.nodata_value,
                write_scaled_result=write_scaled_result
            )

            disaggregation_engine.run()
        
    def beneficiaries_within_cost(self, mode: str = 'ocv_filter2d') -> None:  
        """Determine number of beneficiaries within cost windows.

        :param mode: Method to perform moving window operation. One of 'generic_filter', 'convolve', or 'ocv_filter2d'. Defaults to 'ocv_filter2d', defaults to 'ocv_filter2d'.
        :type mode: str, optional
        """      
        dtype_to_use = np.int32
        
        self.printStepInfo("Determining beneficiaries within costs")


        # read input data
        disaggregated_population_reader, mtx_disaggregated_population = self._get_file("DEMAND/disaggregated_population.tif", [self.nodata_value])                        
        mtx_disaggregated_population[mtx_disaggregated_population == self.nodata_value] = 0
        
        mtx_clumps, clump_nodata_mask = self._get_clumps()
        clump_slices = ndimage.find_objects(mtx_clumps.astype(np.int64))        
        
        step_count = len(self.cost_thresholds) * len(clump_slices)
        current_task = self._new_task("[white]Determining beneficiaries", total=step_count)
        
        with self.progress:
            for c in self.cost_thresholds:

                mtx_pop_within_cost = self._sum_values_in_kernel(
                    mtx_source=mtx_disaggregated_population.astype(dtype_to_use),
                    mtx_clumps=mtx_clumps,
                    clump_slices=clump_slices,
                    cost=c,
                    mode=mode,
                    progress_task=current_task
                )

                # export current beneficiaries within cost
                mtx_pop_within_cost[clump_nodata_mask] = self.nodata_value
                self._write_file(f"DEMAND/beneficiaries_within_cost_{c}.tif", mtx_pop_within_cost, self._get_metadata(dtype_to_use, self.nodata_value))
                del mtx_pop_within_cost
                
        
        # subsequently, we need to order costs and subtract from the larger cost windows the population in the smaller cost windows
        # this is to create layers for population within a specific cost-interval, in order to avoid a double counting of population.
        # before, averaging beneficiaries within cost overestimated population, as larger cost windows included the pop of smaller cost windows, 
        # hence double-counting pop within closer range
        # this should be avoided.
        current_template = Template("DEMAND/beneficiaries_within_cost_range_${cost}.tif")
        self._get_cost_range_differences(self._get_beneficiaries_for_cost, current_template)

        # done
        self.taskProgressReportStepCompleted()

    def average_beneficiaries_across_cost(self, cost_weights: Dict[float, float] = None, write_non_weighted_result: bool = True, write_scaled_result: bool = True) -> None:
        """Determine the number of potential beneficiaries, averaged across cost thresholds. 

        :param cost_weights: Dictionary of cost weights, where keys refer to cost thresholds, and values to weights. If specified, weighted total supply will be determined, defaults to None.
        :type cost_weights: Dict[float, float], optional
        :param write_non_weighted_result: Indicates if non-weighted total supply be computed, defaults to True
        :type write_non_weighted_result: bool, optional
        :param write_scaled_result: Indicates if min-max-scaled values should be written as separate outputs, defaults to True
        :type write_scaled_result: bool, optional
        """        

        self.printStepInfo("Averaging beneficiaries across costs")
        step_count = len(self.cost_thresholds)
        current_task = self._new_task("[white]Averaging beneficiaries", total=step_count)

        with self.progress as p:

            # get clump nodata            
            mtx_clumps, clump_nodata_mask = self._get_clumps()

            # result raster
            if write_non_weighted_result:
                averaged_beneficiaries = self._get_matrix(0, self._get_shape(), np.float64)
            if cost_weights is not None:
                cost_weighted_averaged_beneficiaries = self._get_matrix(0, self._get_shape(), np.float64)

            # iterate over cost thresholds and aggregate cost-specific beneficiaries into result
            for c in self.cost_thresholds:

                if write_non_weighted_result:
                    mtx_current_pop = self._get_beneficiaries_for_cost(c, return_cost_window_difference=False)
                    averaged_beneficiaries += mtx_current_pop
                if cost_weights is not None:
                    mtx_current_pop = self._get_beneficiaries_for_cost(c, return_cost_window_difference=True)
                    cost_weighted_averaged_beneficiaries += (mtx_current_pop * cost_weights[c])
                
                p.update(current_task, advance=1)

            # export averaged diversity grids
            if write_non_weighted_result:
                averaged_beneficiaries = averaged_beneficiaries / len(self.cost_thresholds)
                averaged_beneficiaries[clump_nodata_mask] = self.nodata_value
                self._write_file("INDICATORS/non_weighted_avg_population.tif", averaged_beneficiaries, self._get_metadata(np.float64, self.nodata_value))
                
                # if write_scaled_result:
                #     # apply min-max scaling
                #     scaler = MinMaxScaler()
                #     average_pop = scaler.fit_transform(average_pop.reshape([-1,1]))
                #     self._write_dataset('INDICATORS/scaled_non_weighted_avg_population.tif', average_pop.reshape(self.lsm_mtx.shape))

            if cost_weights is not None:
                cost_weighted_averaged_beneficiaries = cost_weighted_averaged_beneficiaries / sum(cost_weights.values())
                cost_weighted_averaged_beneficiaries[clump_nodata_mask] = self.nodata_value
                self._write_file("INDICATORS/cost_weighted_avg_population.tif", cost_weighted_averaged_beneficiaries, self._get_metadata(np.float64, self.nodata_value))
                
                # if write_scaled_result:
                #     # apply min-max scaling
                #     scaler = MinMaxScaler()
                #     cost_weighted_average_pop = scaler.fit_transform(cost_weighted_average_pop.reshape([-1,1]))
                #     self._write_dataset('INDICATORS/scaled_cost_weighted_avg_population.tif', cost_weighted_average_pop.reshape(self.lsm_mtx.shape))

        # done
        self.taskProgressReportStepCompleted()

#endregion

#region cost

    def _compute_proximity_raster_for_land_use(self, rst_clumps, clump_slices, lu, mode, current_task) -> np.ndarray:

        # target raster
        lu_dr = self._get_matrix(0, self._get_shape(), np.float32)

        # for proximities, edge or patch does not matter basically - if within boundaries of edge class, proximity should still be 0, as distance to edge only matters from outside.
        # however, as prox are determined at clump level, we need to add edge to mask in order to have potential buffered edges included, so that in case of barrier classes,
        # we get a proper depiction of prox
        
        lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
        src_lu_mtx_reader, src_lu_mtx = self._get_file(f'MASKS/mask_{lu}.tif', [self.nodata_value])
        src_lu_mtx[src_lu_mtx == self.nodata_value] = 0

        if lu_type == "edge":                    
            mtx_edge_reader, mtx_edge = self._get_file(f'MASKS/edges_{lu}.tif', [self.nodata_value])
            src_lu_mtx[mtx_edge == 1] = 1
                        
        # add here support for clumping, i.e., determine proximities only to available opportunities within each clump
        # check how to integrate that into the dr raster in the end
        # now operate over clumps, in order to safe some computational time
        for patch_idx in range(len(clump_slices)):
            obj_slice = clump_slices[patch_idx]
            obj_label = patch_idx + 1

            # get slice from land-use mask
            sliced_lu_mtx = src_lu_mtx[obj_slice].copy() 
            sliced_clump_mtx = rst_clumps[obj_slice]
    
            # properly mask out current object
            obj_mask = np.isin(sliced_clump_mtx, [obj_label], invert=False)
            sliced_lu_mtx[~obj_mask] = 0

            # check if we actually have opportunity in reach in current clump slice:
            if np.sum(sliced_lu_mtx) > 0:
                
                if mode == 'dr':
                    # now all pixels outside of clump should be zeroed, and we can determine proximity on the subset of the full raster
                    sliced_dr = dr.DistanceRaster(sliced_lu_mtx, progress_bar=False)
                    sliced_dr = sliced_dr.dist_array
                elif mode == 'xr':
                    n, m = sliced_lu_mtx.shape
                    xr_rst = xr.DataArray(sliced_lu_mtx, dims=['y', 'x'], name='raster')
                    xr_rst['y'] = np.arange(n)[::-1]
                    xr_rst['x'] = np.arange(m)
                    sliced_dr = proximity(xr_rst, target_values = [1]).to_numpy()
                
                # proximities should only be written to clump object
                sliced_dr[~obj_mask] = 0
                lu_dr[obj_slice] += sliced_dr
                                        
            self.progress.update(current_task, advance=1)
        
        del src_lu_mtx
        return lu_dr

    def compute_proximity_rasters(self, mode: str = 'xr', lu_classes: List[int] = None, assess_builtup: bool = False) -> None:
        """Generate proximity rasters to land-use classes based on identified clumps.

        :param mode: Method used to compute proximity matrix. Either 'dr' or 'xr', defaults to 'xr'
        :type mode: str, optional
        :param lu_classes: List of integers, i.e., land-use classes to assess, defaults to None
        :type lu_classes: List[int], optional
        :param assess_builtup: Assesses proximities to built-up, if true, defaults to False
        :type assess_builtup: bool, optional
        """        

        self.printStepInfo("Computing distance rasters")
        # determine proximity outward from relevant lu classes, including built-up
        classes_for_proximity_calculation = (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge) if lu_classes is None else lu_classes

        # import the clumps raster
        # detect slices here, to avoid having to recall detect_clumps each time we want to do proximity computations
        rst_clumps, clump_nodata_mask = self._get_clumps()
        clump_slices = ndimage.find_objects(rst_clumps.astype(np.int64))        
        
        step_count = len(classes_for_proximity_calculation) * len(clump_slices)
        current_task = self._new_task("[white]Computing distance rasters", total=step_count)

        # iterate over classes and clumps
        with self.progress:
            for lu in classes_for_proximity_calculation:
                # make computation
                lu_dr = self._compute_proximity_raster_for_land_use(rst_clumps, clump_slices, lu, mode, current_task)
                # mask and export lu prox
                lu_dr[clump_nodata_mask] = self.nodata_value
                self._write_file(f"PROXIMITY/dr_{lu}.tif", lu_dr, self._get_metadata(np.float32, self.nodata_value))                
                # clean up
                del lu_dr
               
        if assess_builtup:
            
            step_count = len(self.lu_classes_builtup) * len(clump_slices)
            current_task = self._new_task("[white]Computing distance rasters to built-up", total=step_count)
            with self.progress:
                for lu in self.lu_classes_builtup:
                     # make computation
                    lu_dr = self._compute_proximity_raster_for_land_use(rst_clumps, clump_slices, lu, mode, current_task)
                    # mask and export lu prox
                    lu_dr[clump_nodata_mask] = self.nodata_value
                    self._write_file(f"PROXIMITY/dr_{lu}.tif", lu_dr, self._get_metadata(np.float32, self.nodata_value))                    
                    # clean up
                    del lu_dr
 
        # done
        self.taskProgressReportStepCompleted()

    def cost_to_closest(self, lu_classes = None) -> None:
        
        # several assumptions need to be considered when computing costs:
        # the output of distances is...
        #   0 outside of clumps, as these are nodata areas (=nodata)
        # > 0 inside of clumps, when lu within clump (=proximity)
        #   0 inside of clumps, within lu of interest (=presence)
        #   0 inside of clumps, if lu not within clump (=nodata)  

        self.printStepInfo("Assessing cost to closest")
        included_lu_classes = lu_classes if lu_classes is not None else self.lu_classes_recreation_patch + self.lu_classes_recreation_edge

        # we require clumps for masking
        mtx_clumps, clump_nodata_mask = self._get_clumps()
        clump_slices = ndimage.find_objects(mtx_clumps.astype(np.int64))
        
        step_count = len(included_lu_classes) * len(clump_slices)
        current_task = self._new_task("[white]Assessing cost to closest", total=step_count)

        with self.progress as p:
            
            # iterate over land-uses
            for lu in included_lu_classes:
                
                # store final result
                mtx_out = self._get_matrix(0, self._get_shape(), np.float32)
                
                # get relevant lu-specific datasets
                # complete cost raster
                mtx_proximity_to_lu = self._get_proximity_raster_for_lu(lu) 
                
                # complete mask raster
                mtx_lu_mask = self._get_land_use_class_mask(lu)

                # iterate over patches, and for each patch, determine whether sum of mask is 0 (then all 0 costs are nodata)
                # or whether sum of mask > 0, then 0 are actual within-lu costs, and remaining values should be > 0 for the current lu
                for patch_idx in range(len(clump_slices)):
                    
                    obj_slice = clump_slices[patch_idx]
                    obj_label = patch_idx + 1

                    # get slice from land-use mask
                    sliced_lu_mask = mtx_lu_mask[obj_slice].copy() 
                    sliced_mtx_clumps = mtx_clumps[obj_slice]

                    # properly mask out current object
                    obj_mask = np.isin(sliced_mtx_clumps, [obj_label], invert=False)
                    sliced_lu_mask[~obj_mask] = 0

                    # now the sliced mask is 0 outside of the clump, and 0 or 1 within the clump
                    # hence, if the sum of sliced mask is now >0, we need to continue 
                    # with proximities. Otherwise, proximities = 0 are equal to nodata 
                    # as lu not within clump. in that case, it does not count toward the average 
                    if np.sum(sliced_lu_mask) > 0:
                        
                        # write out proximities
                        mtx_sliced_proximity_to_lu = mtx_proximity_to_lu[obj_slice].copy()                        
                        # add one to allow re-masking with 0. we will subtract that later for the completed matrix
                        np.add(mtx_sliced_proximity_to_lu, 1, out=mtx_sliced_proximity_to_lu)                        
                        
                        mtx_sliced_proximity_to_lu[~obj_mask] = 0                      
                        mtx_out[obj_slice] += mtx_sliced_proximity_to_lu

                    p.update(current_task, advance=1)


                # done iterating over patches
                del mtx_lu_mask
                del mtx_proximity_to_lu

                # now apply nodata value to all values that are 0, as we shifted all proximities by +1
                mtx_out[mtx_out <= 0] = self.nodata_value
                np.subtract(mtx_out, 1, out=mtx_out, where=mtx_out > 0)

                # export mtx_out for current lu
                self._write_file(f'COSTS/minimum_cost_{lu}.tif', mtx_out, self._get_metadata(np.float32, self.nodata_value))

        # done
        self.taskProgressReportStepCompleted()

    def minimum_cost_to_closest(self, lu_classes = None, write_scaled_result: bool = True) -> None:
        
        self.printStepCompleteInfo("Assessing minimum cost to closest")

        included_lu_classes = lu_classes if lu_classes is not None else (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge)
        
        step_count = len(included_lu_classes)
        current_task = self._new_task("[white]Assessing minimum cost to closest", total=step_count)

        # make result layer
        high_val = 9999999
        
        mtx_min_cost = self._get_matrix(high_val, self._get_shape(), np.float32)
        mtx_min_type = self._get_matrix(0, self._get_shape(), np.float32)

        with self.progress as p:
            for lu in included_lu_classes:

                mtx_proximity = self._get_minimum_cost_for_lu(lu) 
                
                # in mtx_proximity, we have any form of actual distance as values >= 0, and
                # all other pixels as nodata_value (by default, -9999)
                condition = ((mtx_proximity < mtx_min_cost) & (mtx_proximity != self.nodata_value))

                mtx_min_cost[condition] = mtx_proximity[condition]                
                mtx_min_type = np.where(condition, lu, mtx_min_type)

                p.update(current_task, advance=1)

            mtx_min_cost[mtx_min_cost == high_val] = self.nodata_value
            
            # mask and export layers
            self._write_file('INDICATORS/non_weighted_minimum_cost.tif', mtx_min_cost, self._get_metadata(np.float32, self.nodata_value))
            
            mtx_min_type[mtx_min_type == 0] = self.nodata_value
            self._write_file('INDICATORS/minimum_opportunity_type.tif', mtx_min_type, self._get_metadata(np.float32, self.nodata_value))

        # done
        self.taskProgressReportStepCompleted()

    def average_cost_to_closest(self, lu_classes = None, distance_threshold: float = -1, write_scaled_result: bool = True) -> None:
        
        # TODO: Add lu-based weighting

        self.printStepInfo("Assessing average cost to closest")
        included_lu_classes = lu_classes if lu_classes is not None else self.lu_classes_recreation_patch + self.lu_classes_recreation_edge
                
        step_count = len(included_lu_classes)
        current_task = self._new_task("[white]Averaging cost to closest", total=step_count)

        # raster for average result
        mtx_average_cost = self._get_matrix(0, self._get_shape(), np.float64)
        mtx_lu_cost_count_considered = self._get_matrix(0, self._get_shape(), np.float64)    
        
        if distance_threshold > 0:
            print(f"{Fore.YELLOW}{Style.BRIGHT}Masking costs > {distance_threshold} units{Style.RESET_ALL}")                

        with self.progress as p:

            # get clumps raster for masking
            rst_clumps, clump_nodata_mask = self._get_clumps()

            # now operate over clumps, in order to safe some computational time
            for lu in included_lu_classes:
                
                # get relevant lu-specific datasets
                # complete cost raster
                mtx_lu_prox = self._get_minimum_cost_for_lu(lu)
                
                # the mtx_lu_prox raster contains values of nodata if no lu in clump/outside of clump, 
                # or >= 0 when cost is within or towards lu
                # by default, the nodata value is -9999 and it can be used for masking of the raster
                # or, we may resort to using >= 0
                
                # if we require cost masking by distance thresholds > 0, 
                # mask inputs here
                if distance_threshold > 0:
                    mtx_lu_prox[mtx_lu_prox > distance_threshold] = self.nodata_value

                np.add(mtx_average_cost, mtx_lu_prox, out=mtx_average_cost, where=mtx_lu_prox != self.nodata_value)
                np.add(mtx_lu_cost_count_considered, 1, out=mtx_lu_cost_count_considered, where=mtx_lu_prox != self.nodata_value)
           
                p.update(current_task, advance=1)

                del mtx_lu_prox

        # export average cost grid
        # prior, determine actual average. here, consider per each pixel the number of grids added.
        # self._write_dataset('COSTS/raw_sum_of_cost.tif', mtx_average_cost, mask_nodata=False, custom_metadata=custom_meta)

        # properly mask nodata values from clumps mask
        mtx_lu_cost_count_considered[clump_nodata_mask] = self.nodata_value
        self._write_file('COSTS/cost_count.tif', mtx_lu_cost_count_considered, self._get_metadata(np.float64, self.nodata_value))
                
        np.divide(mtx_average_cost, mtx_lu_cost_count_considered, out=mtx_average_cost, where=mtx_lu_cost_count_considered > 0)     
        
        # we should now also be able to reset cells with nodata values to the actual nodata_value
        # mtx_average_cost[mtx_lu_cost_count_considered < 0] = self.nodata_value
        mtx_average_cost[clump_nodata_mask] = self.nodata_value
        self._write_file('INDICATORS/non_weighted_avg_cost.tif', mtx_average_cost, self._get_metadata(np.float64, self.nodata_value))
        
        # if write_scaled_result:
        #     # apply min-max scaling
        #     scaler = MinMaxScaler()
        #     mtx_average_cost = 1-scaler.fit_transform(mtx_average_cost.reshape([-1,1]))
        #     self._write_dataset('INDICATORS/scaled_non_weighted_avg_cost.tif', mtx_average_cost.reshape(self.lsm_mtx.shape), mask_nodata=False)

        del mtx_average_cost
        del mtx_lu_cost_count_considered

        # done
        self.taskProgressReportStepCompleted()

#endregion

#region flow


    def detect_land_use_patches(self, lu_classes: List[int] = None, contiguity: Contiguity = Contiguity.Queen):
        """Identify contiguous patches of land-uses.

        :param lu_classes: List of land-use classes to assess, defaults to None. If None, all patch and edge classes will be considered.
        :type lu_classes: List[int], optional
        :param contiguity: Specifies whether to use Queen or Rook contiguity, defaults to Queen.
        :type contiguity: recreat.Contiguity
        """
        self.printStepInfo("Detecting land-uses patches")

        lu_classes = lu_classes if lu_classes is not None else (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge)

        step_count = len(lu_classes)
        current_task = self._new_task("[white]Determine patches", total=step_count)

        with self.progress as p:

            for lu in lu_classes:   

                # get class mask
                mtx_lu_class_mask = self._get_land_use_class_mask(lu)
                nr_clumps, mtx_clumps = self._detect_clumps_in_raster(mtx_lu_class_mask, barrier_classes=[0], contiguity=contiguity)
                
                # write result
                self._write_file(f'CLUMPS_LU/clumps_{lu}.tif', mtx_clumps, self._get_metadata(np.int32, self.nodata_value))
                p.update(current_task, advance=1)

        self.printStepCompleteInfo()

    def class_patch_flow(self) -> None:
        """Determine flow to patches of recreational land-use classes.
        """

        self.printStepInfo("Determine patch flow")        

        lu_classes = (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge) 
        
        step_count = len(lu_classes) * len(self.cost_thresholds)
        current_task = self._new_task("[white]Determine patch flow", step_count)

        with self.progress as p:
            pass

        self.printStepCompleteInfo()






    def class_flow(self) -> None:
        """Determine the total number of potential beneficiaries (flow to given land-use classes) as the sum of total population, within cost thresholds.
        """

        self.printStepInfo("Determine class flow")        
        
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch))
        current_task = self._new_task("[white]Determine class-based flows within cost", step_count)

        with self.progress as p:            
            
            # get clump data
            mtx_clumps, clump_nodata_mask = self._get_clumps()

            for c in self.cost_thresholds:
                
                mtx_pop = self._get_beneficiaries_for_cost(c)

                for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):                
                    
                    # determine source of list
                    lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
                    mtx_lu = self._get_land_use_class_mask(lu)
                    
                    mtx_res = self._get_matrix(0, self._get_shape(), np.int32)
                    np.multiply(mtx_lu, mtx_pop, out=mtx_res)                                        

                    # mask nodata areas and write result
                    mtx_res[clump_nodata_mask] = self.nodata_value
                    outfile_name = f"FLOWS/flow_class_{lu}_cost_{c}.tif" if lu_type == 'patch' else f"FLOWS/flow_edge_class_{lu}_cost_{c}.tif"
                    self._write_file(outfile_name, mtx_res, self._get_metadata(np.int32, self.nodata_value))
                    
                    del mtx_res
                    del mtx_lu

                    p.update(current_task, advance=1)

                del mtx_pop

        # done
        self.taskProgressReportStepCompleted()
    
    def aggregate_class_flow(self) -> None:
        """Aggregate flow for a given cost threshold across all land-use classes.

        :param cost: Cost threshold
        :type cost: int
        """
        self.printStepInfo("Aggregating flow within cost")        

        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge)) 
        current_task = self._new_task("[white]Averaging flow across costs", total=step_count)

        with self.progress as p:

            mtx_clumps, clump_nodata_mask = self._get_clumps()            
            for c in self.cost_thresholds:
                
                mtx_flow_at_cost = self._get_aggregate_class_flow_for_cost(c)
                mtx_flow_at_cost[clump_nodata_mask] = self.nodata_value
                self._write_file(f'FLOWS/flow_for_cost_{c}.tif', mtx_flow_at_cost, self._get_metadata(np.int32, self.nodata_value))
                p.update(current_task, advance=1)

        # additionally, determine differences across cost thresholds
        current_template = Template("FLOWS/flow_within_cost_range_${cost}.tif")
        self._get_cost_range_differences(self._get_flow_for_cost, current_template)

        self.printStepCompleteInfo()


    def _get_aggregate_class_flow_for_cost(self, cost):

        mtx_flow_for_current_cost = self._get_matrix(0, self._get_shape(), np.int32)
        # iterate over cost thresholds and lu classes                
        for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):  
            mtx_current_flow = self._get_flow_for_land_use_class_and_cost(lu, cost)
            mtx_flow_for_current_cost += mtx_current_flow 
        
        return mtx_flow_for_current_cost
            

    def _get_cost_range_differences(self, get_data_method, outfile_template: Template, **kwargs) -> None:
        
        self.printStepInfo("Determining cost-range differences")

        step_count = len(self.cost_thresholds)
        current_task = self._new_task("[white]Iterating cost ranges", total=step_count)
        
        rst_clumps, clump_nodata_mask = self._get_clumps()

        with self.progress:
            
            # assert order from lowest to highest cost
            sorted_costs = sorted(self.cost_thresholds)
            
            # write lowest range directly
            mtx_lower_range = get_data_method(cost=sorted_costs[0], **kwargs)
            mtx_lower_range[clump_nodata_mask] = self.nodata_value
            
            out_filename = outfile_template.substitute(cost=f"{sorted_costs[0]}") 
            self._write_file(out_filename, mtx_lower_range.astype(np.float64), self._get_metadata(np.float64, self.nodata_value))
            del mtx_lower_range
            self.progress.update(current_task, advance=1)

            for i in range(1,len(sorted_costs)):
                
                mtx_lower_range = get_data_method(cost=sorted_costs[i-1], **kwargs) 
                mtx_current_cost = get_data_method(cost=sorted_costs[i], **kwargs)

                mtx_flow_in_cost_range = self._get_matrix(0, self._get_shape(), np.float64)
                np.subtract(mtx_current_cost, mtx_lower_range, out=mtx_flow_in_cost_range)

                # mask nodata and export
                mtx_flow_in_cost_range[clump_nodata_mask] = self.nodata_value
                out_filename = outfile_template.substitute(cost=f"{sorted_costs[i]}") 
                self._write_file(out_filename, mtx_flow_in_cost_range, self._get_metadata(np.float64, self.nodata_value))
                self.progress.update(current_task, advance=1)

    def average_flow_across_cost(self, cost_weights: Dict[float, float] = None, write_non_weighted_result: bool = True, write_scaled_result: bool = True):
        """Determine the number of potential beneficiaries in terms of flow to (recreational) land-use classes, averaged across cost thresholds.

        :param cost_weights: Dictionary of cost weights, where keys refer to cost thresholds, and values to weights. If specified, weighted total supply will be determined, defaults to None
        :type cost_weights: Dict[float, float], optional
        :param write_non_weighted_result: Indicates if non-weighted total supply be computed, defaults to True
        :type write_non_weighted_result: bool, optional
        :param write_scaled_result: Indicates if min-max-scaled values should be written as separate outputs, defaults to True
        :type write_scaled_result: bool, optional
        """        

        self.printStepInfo("Averaging flow across costs")    

        step_count = len(self.cost_thresholds)
        current_task = self._new_task("[white]Averaging flow across costs", total=step_count)

        with self.progress as p:

            # get clump nodata mask
            mtx_clumps, clump_nodata_mask = self._get_clumps()

            # result grids for integrating averaged flows
            if write_non_weighted_result:
                mtx_averaged_flow = self._get_matrix(0, self._get_shape(), np.float64)
            if cost_weights is not None:
                mtx_cost_weighted_average_flow = self._get_matrix(0, self._get_shape(), np.float64)

            # here, we need to iterate over cost windows only, as flow has previously been aggregate at the level of cost windows
            # so we do not need to aggregate flow across land-uses
            for c in self.cost_thresholds:


                if write_non_weighted_result:
                    mtx_flow_in_cost = self._get_flow_for_cost(c, return_cost_window_difference=False)
                    mtx_averaged_flow += mtx_flow_in_cost
                if cost_weights is not None:
                    mtx_flow_in_cost = self._get_flow_for_cost(c, return_cost_window_difference=True)                    
                    mtx_cost_weighted_average_flow += (mtx_flow_in_cost * cost_weights[c])

                p.update(current_task, advance=1)
                
            # export integrated grids
            if write_non_weighted_result:
                mtx_averaged_flow[clump_nodata_mask] = self.nodata_value
                self._write_file("INDICATORS/non_weighted_avg_flow.tif", mtx_averaged_flow, self._get_metadata(np.float64, self.nodata_value))
                # if write_scaled_result:
                #     # apply min-max scaling
                #     scaler = MinMaxScaler()
                #     integrated_average_flow = scaler.fit_transform(integrated_average_flow.reshape([-1,1]))
                #     self._write_dataset('FLOWS/scaled_integrated_avg_flow.tif', integrated_average_flow.reshape(self.lsm_mtx.shape))
            
            if cost_weights is not None:
                # scale cost-weighted matrix by sum of weights
                mtx_cost_weighted_average_flow = mtx_cost_weighted_average_flow /  sum(cost_weights.values())
                mtx_cost_weighted_average_flow[clump_nodata_mask] = self.nodata_value
                self._write_file("INDICATORS/cost_weighted_avg_flow.tif", mtx_cost_weighted_average_flow, self._get_metadata(np.float64, self.nodata_value))
                # if write_scaled_result:
                #     # apply min-max scaling
                #     scaler = MinMaxScaler()
                #     integrated_cost_weighted_average_flow = scaler.fit_transform(integrated_cost_weighted_average_flow.reshape([-1,1]))
                #     self._write_dataset('FLOWS/scaled_integrated_cost_weighted_avg_flow.tif', integrated_cost_weighted_average_flow.reshape(self.lsm_mtx.shape))

        # done
        self.taskProgressReportStepCompleted()

#endregion

#region quality of access estimation


    def make_quality_of_access_map(self, lu_classes: List[int], accessibility_ranges: List[List[int]]):
        """This function generates a mapping of potential accessibility (i.e., presence within defined costs) to specified land-use classes. 
           For example, this could correspond to a mapping of accessibility to (presence of) any forest-related classes in near, intermediate, 
           and remote proximity in terms of comparatively lower, moderate, or higher costs. Presence or absence of a given land-use class 
           is determined as the difference in supply within a specific cost compared to the immediate lower cost range.  

        :param lu_classes: Land-use classes to include in the mapping. The classes will be treated integratively, not separately.  
        :type lu_classes: List[int]
        :param accessibility_ranges: List of lists cost ranges to assess. A range may comprise a single or multiple cost thresholds. 
        :type accessibility_ranges: List[List[int]]
        """
        quality_mappings: List[Tuple[List[int], float]] = []
        mtx_clumps, clump_nodata_mask = self._get_clumps()

        # automatically scale integer over cost ranges, which need to be provided in ascending order from best to worst.
        i: int = 1
        for cost_range in reversed(accessibility_ranges):
            quality_mappings.append((cost_range, i))
            i = i * 10

        print(quality_mappings)

        # iterate over cost ranges and make differences: supply should be >0 to be present; 
        # a difference of 0 to previous cost ranges would indicate no additional supply; a difference > 0 would indicate additional supply in this cost range
        # make difference rasters

        mtx_out = np.zeros(self._get_shape(), np.int32)

        # iterate over land-use classes
        for lu in lu_classes:
            current_template = Template("SUPPLY/totalsupply_" + str(lu) + "_within_cost_range_${cost}.tif")
            self._get_cost_range_differences(self._get_supply_for_land_use_class_and_cost, current_template, lu=lu)

        # iterate over cost_ranges
        for cost_range in quality_mappings:                
            
            mtx_step = np.zeros(self._get_shape(), np.int32)
            current_scaling_factor = cost_range[1]

            # iterate over classes considered jointly
            for lu in lu_classes:
                # iterate over cost threshold
                for c in cost_range[0]:
                    c_data = self._get_supply_for_land_use_class_and_cost(lu, c, return_cost_window_difference=True)
                    c_data[c_data > 0] = 1
                    np.add(mtx_step, c_data.astype(np.int32), out=mtx_step)

            # everything in this step added. remask to 0/1 schema and apply scaling factor
            mtx_step[mtx_step > 0] = current_scaling_factor
            np.add(mtx_out, mtx_step, out=mtx_out)

        mtx_out[clump_nodata_mask] = self.nodata_value
        self._write_file(os.path.join('INDICATORS', f'access_to_{"_".join([str(k) for k in lu_classes])}.tif'), mtx_out, self._get_metadata(np.int32, self.nodata_value))           

#endregion



#region detailed flow base data generation 

    def patch_properties_tables(self, lu_classes: List[int] = None):
        """This function computes land-use patch properties, i.e., number of pixels and respective patch size, depending on land-use raster resolution.
        This is required of patch sizes should be considered in the estimation of detailed flow.

        :param lu_classes: List of land-use classes to include in the assessment, defaults to None
        :type lu_classes: List[int], optional
        """

        self.printStepInfo("Determine patch properties")
        lu_classes = lu_classes if lu_classes is not None else (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge)

        step_count = len(lu_classes)
        current_task = self._new_task("[white]Estimating properties", total=step_count)

        # read resolution as this will be used to compute area (should be in sqm)
        resolution = self.land_use_map_reader.res

        with self.progress as p:

            # iterate over lu classes, determine patch properties using regionprops_table method in skimage,
            # and store result as parquet file
            for lu in lu_classes:

                mtx_lu_patches = self._get_lu_patches(lu)

                tbl_lu_patch_props = measure.regionprops_table(mtx_lu_patches, properties=('label', 'num_pixels'))
                patch_df = pd.DataFrame(tbl_lu_patch_props)
                patch_df['land_use'] = lu
                patch_df['patch_label'] = patch_df['label']                
                patch_df['area'] = patch_df['num_pixels'] * (resolution[0]**2)
                patch_df.drop(columns=['label'], inplace=True)

                out_path = self.get_file_path(os.path.join('CLUMPS_LU', f'table_{lu}.pqt'))
                patch_df.to_parquet(out_path, index=False)

                p.update(current_task, advance=1)

        # done
        self.printStepCompleteInfo()
        

    def lu_accessibility_tables(self, lu_classes: List[int] = None):
        """This function determines, for each land-use class included in the assessment, and for each clump, the patches and associated costs within each populated pixel. 
        This method will write, per land-use class, a parquet file containing clump labels, populated pixel coordinates with respect to clump, land-use class, and patch labels 
        as variables. These data are the basic input data for the modelling of detailed flow.   

        :param lu_classes: Land-use classes to include in the assessment, defaults to None
        :type lu_classes: List[int], optional
        """

        self.printStepInfo("Determine flow variables")

        # determine classes to iterate
        lu_classes = lu_classes if lu_classes is not None else (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge)

        step_count = len(lu_classes)
        current_task = self._new_task("[white]Determine flow variables", total=step_count)

        with self.progress as p:

            # iterate over classes and store as parquet files
            for lu in lu_classes:
                
                df_for_class = self._lu_accessibility_and_properties(lu=lu)
                out_path = self.get_file_path(os.path.join('FLOWS_DF', f'flow_df_{lu}.pqt'))
                df_for_class.to_parquet(out_path, index=False)
                
                p.update(current_task, advance=1)
            
        # done    
        self.printStepCompleteInfo()

 
    def _lu_accessibility_and_properties(self, lu: int) -> pd.DataFrame:

        results = []

        mtx_clumps, clump_nodata_mask = self._get_clumps()
        mtx_clumps[mtx_clumps == self.nodata_value] = 0
        clump_slices = ndimage.find_objects(mtx_clumps.astype(np.int64))
        
        # get relevant data
        mtx_beneficiaries = self._get_disaggregated_population()
        mtx_lu_patches = self._get_lu_patches(lu)

        step_count = len(clump_slices)
        current_task = self._new_subtask(f"[white]Processing class {lu}", total=step_count)


        def process_clump(patch_idx):    
            
            # clump results will store all patches within costs and corresponding pop of origin cell including coordinates of cell as a reference
            clump_results = {}

            obj_slice = clump_slices[patch_idx]
            obj_label = patch_idx + 1

            # get slice from land-use mask
            sliced_mtx_source = mtx_lu_patches[obj_slice].copy()
            sliced_mtx_demand = mtx_beneficiaries[obj_slice].copy()
            sliced_mtx_clumps = mtx_clumps[obj_slice]

            # properly mask out current object
            obj_mask = np.isin(sliced_mtx_clumps, [obj_label], invert=False)

            sliced_mtx_source[~obj_mask] = 0
            sliced_mtx_demand[~obj_mask] = 0

            rows, cols = sliced_mtx_demand.shape

            # sliced_mtx_source now contains properly masked lu patches within the current clump
            # sliced_mtx_demand now contains properly masked demand (disaggr. pop.) within the current clump

            # iterate over the demand raster, i.e., each value in demand will be assigned to 
            # patches, depending on a certain distrubution function                
            for row in range(rows):
                for col in range(cols):

                    # we also need to know how many land-uses are within reach, as we have to divide the number of
                    # beneficiaries to be distributed in this cost to each land

                    current_population_value = sliced_mtx_demand[row,col]
                    if current_population_value == 0:
                        continue # nothing to distribute here!

                    for c in self.cost_thresholds:                                                        

                        kernel_size = c
                        row_start = max(0, row - kernel_size // 2)
                        row_end = min(rows, row + kernel_size // 2 + 1)
                        col_start = max(0, col - kernel_size // 2)
                        col_end = min(cols, col + kernel_size // 2 + 1)

                        patch_labels_in_kernel = sliced_mtx_source[row_start:row_end, col_start:col_end]
                        # get the unique values that are not backgground!
                        unique_patch_labels = np.unique(patch_labels_in_kernel)

                        # get properties of patches
                        for patch_label in unique_patch_labels:

                            if patch_label == 0:
                                # 0 is the background
                                continue

                            # include the current pixel in the resultset if not yet in there. 
                            contains_pixel = (row,col) in clump_results.keys()
                            if not contains_pixel:
                                clump_results[(row,col)] = {}

                            # add the clump, and indicate whether a clump has been included at lower cost thresholds. 
                            # this will allow for most flexibility to either consider large clumps within distinct costs, or ignore large clumps in higher cost windows.
                            contains_patch = False if not contains_pixel else patch_label in clump_results[(row,col)].keys()                                
                            
                            multiple_observations = 0
                            if contains_patch:
                                multiple_observations = 1

                            # add to results
                            clump_results[(row,col)][patch_label] = [obj_label, row, col, lu, patch_label, current_population_value, c, multiple_observations]

                                
            # done iterating over all pixels of the current clump. if we have a resultset, add this to the final results
            # add clump result to list of results
            flattened_dict = [g for k in clump_results.values() for g in k.values()]
            if len(flattened_dict) > 0:
                results.append(flattened_dict)

            #results.append(clump_results)
            self.progress.update(current_task, advance=1)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_clump, patch_idx) for patch_idx in range(len(clump_slices))]
            concurrent.futures.wait(futures)   

        # flatten results to make a dataframe
        data = [x for cl in results for x in cl]
        df = pd.DataFrame(data, columns=['clump_label', 'row', 'col', 'land_use', 'patch_label', 'orig_pop', 'cost', 'multiple_obs'])

        #done
        # remove task 
        self.progress.remove_task(current_task)
        return df
        

#endregion

#region detailed flow modelling
    
    def model_flow(self, lu_classes: List[int], allocation_method: Callable, outfile_path: str) -> pd.DataFrame:

        self.printStepInfo("Estimating flows")

        # determine classes to iterate
        lu_classes = lu_classes if lu_classes is not None else (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge)

        # result df
        df_estimated_flows: pd.DataFrame = None

        # make dfs with variables
        flow_vars: pd.DataFrame = None
        patch_vars: pd.DataFrame = None

        for lu in lu_classes:
            # read flow
            tmp_path = self.get_file_path(os.path.join("FLOWS_DF", f"flow_df_{lu}.pqt"))
            tmp_df = pd.read_parquet(tmp_path)
            flow_vars = tmp_df if flow_vars is None else pd.concat([flow_vars, tmp_df], ignore_index=True)

            # read patch
            tmp_path = self.get_file_path(os.path.join("CLUMPS_LU", f"table_{lu}.pqt"))
            tmp_df = pd.read_parquet(tmp_path)
            patch_vars = tmp_df if patch_vars is None else pd.concat([patch_vars, tmp_df], ignore_index=True)

        # merge together, free mem
        df_flow_vars = pd.merge(left=flow_vars, right=patch_vars, on=['land_use', 'patch_label'], how='left')
        del flow_vars
        del patch_vars

        # get unique clumps in df; we iterate over those
        unique_clumps = list(df_flow_vars["clump_label"].drop_duplicates())

        step_count = len(unique_clumps)
        current_task = self._new_task("[white]Modelling flow", total=step_count)

        with self.progress as p:

            # parallelize flow
            def get_flow_for_clump(clump_label: int):
                
                df_of_clump = df_flow_vars[df_flow_vars['clump_label'] == clump_label].copy()
                tmp_df = self.allocate_flow_to_patches(df_of_clump, allocation_method)

                # tmp_df is a dataframe containing land_use, patch_label, flow columns
                # re-add clump label and append to final df
                tmp_df['clump_label'] = clump_label
                return tmp_df
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                futures = [executor.submit(get_flow_for_clump, clump_label) for clump_label  in unique_clumps]
                # Iterate over futures as they complete
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()            
                    df_estimated_flows = result if df_estimated_flows is None else pd.concat([df_estimated_flows, result], ignore_index=True)
                    p.update(current_task, advance=1)

        # store as parquet_file
        out_path = self.get_file_path(os.path.join("FLOWS", outfile_path))
        df_estimated_flows.to_parquet(out_path, index=False)

        # done
        self.printStepCompleteInfo()        
        return df_estimated_flows


        
    def allocate_flow_to_patches(self, clump_df: pd.DataFrame, allocation_method) -> pd.DataFrame:

        return_df = None

        # get unique populated places: these are all unique combinations of row and col
        residential_pixels = clump_df[["row", "col"]].drop_duplicates()
        location_count = residential_pixels.shape[0]

        current_task = self._new_subtask("[red]Iterating populated pixels", total=location_count)
        update_int = 0

        def process_location(index, location):
            
            df_of_location = clump_df[((clump_df['row'] == location['row']) & (clump_df['col'] == location['col']))]
            allocation_df = df_of_location[["land_use", "patch_label", "cost", "area", "multiple_obs"]].copy()

            # prepare the result dataframe and add result column accordingly
            allocation_df['flow'] = 0.0
            pop_of_pixel = df_of_location['orig_pop'].max()
            df_current_location = allocation_method(allocation_df, pop_of_pixel)
            df_current_location = df_current_location[['land_use', 'patch_label', 'flow']].copy()

            return df_current_location
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            
            futures = [executor.submit(process_location, index, location) for index, location  in residential_pixels.iterrows()]
            # Iterate over futures as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                return_df = result if return_df is None else pd.concat([return_df, result], ignore_index=True)
                
                update_int += 1
                if update_int > 500:
                    self.progress.update(current_task, advance=update_int)
                    update_int=0

        # done iterating over unique locations in this clump
        # remote the progress
        self.progress.remove_task(current_task)
        # aggregate to final df and return
        return return_df.groupby(['land_use', 'patch_label']).sum().reset_index()

#endregion

#region detailed flow mapping

    def map_flow_from_table(self, path: str, outfile_name: str, lu_classes: List[int] = None) -> None:
        
        self.printStepInfo("Mapping flow from table")

        # import table
        flow_df = pd.read_parquet(path)
        grouped_data = flow_df[['land_use', 'patch_label', 'flow']].groupby(['land_use', 'patch_label'], as_index=False).sum()
        grouped_data = grouped_data[grouped_data['flow'] > 0]

        # determine relevant land uses
        lu_classes = lu_classes if lu_classes is not None else set(grouped_data['land_use'].values.tolist())        
        
        step_count = len(lu_classes)
        current_task = self._new_task("Mapping flow to raster", total=step_count)

        rst_clumps, clump_nodata_mask = self._get_clumps()
        res_mtx = np.zeros(self._get_shape(), np.float32)                
        
        with self.progress as p:
            for lu in lu_classes:
                
                # get portion of data frame relevant for current class
                c_df = grouped_data[(grouped_data['land_use'] == lu)].copy()
                # assert data frame is not empty
                if not c_df.empty:        
                    mtx_lu_patches = self._get_lu_patches(lu)
                    tmp_mtx = np.zeros(self._get_shape(), np.float32)
                    util.map_array(mtx_lu_patches.astype(np.int32), np.array(c_df['patch_label'].values.tolist()), np.array(c_df['flow'].values.tolist()), out=tmp_mtx)
                    np.add(res_mtx, tmp_mtx, out=res_mtx)
            
                p.update(current_task, advance=1)

        # done iterations, write result to disk
        res_mtx[clump_nodata_mask] = self.nodata_value        
        self._write_file(f"INDICATORS/{outfile_name}", res_mtx, self._get_metadata(np.float32, self.nodata_value))
        
        # done
        self.printStepCompleteInfo()



#endregion