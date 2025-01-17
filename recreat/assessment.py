###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################
import os
from os import listdir
from os.path import isfile, join
import ctypes
import platform

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

from .transformations import Transformations
from .disaggregation import SimpleAreaWeightedEngine, DasymetricMappingEngine, DisaggregationMethod
from .exceptions import MethodNotImplemented
from .base import RecreatBase

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
        """Create required subfolders for raster files in the current scenario folder.
        """
        # create directories, if needed
        dirs_required = ['DEMAND', 'MASKS', 'SUPPLY', 'INDICATORS', 'TMP', 'FLOWS', 'CLUMPS_LU', 'PROX', 'COSTS', 'DIVERSITY', 'BASE']
        for d in dirs_required:
            current_path = f"{self.data_path}/{self.root_path}/{d}"
            if not os.path.exists(current_path):
                os.makedirs(current_path)


    def set_params(self, param_name: str, param_value: any) -> None:

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

    def _get_supply_for_land_use_class_and_cost(self, lu: int, cost: int) -> np.ndarray:        
        lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"        
        filename = f"SUPPLY/totalsupply_class_{lu}_cost_{cost}_clumped.tif" if lu_type == 'patch' else f"SUPPLY/totalsupply_edge_class_{lu}_cost_{cost}_clumped.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)
    
    def _get_land_use_class_mask(self, lu: int) -> np.ndarray:        
        lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
        filename = f"MASKS/mask_{lu}.tif" if lu_type == 'patch' else f"MASKS/edges_{lu}.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)
    
    def _get_diversity_for_cost(self, cost: int) -> np.ndarray:
        filename = f"DIVERSITY/diversity_cost_{cost}.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)
    
    def _get_beneficiaries_for_cost(self, cost: int, return_cost_window_difference: bool = False) -> np.ndarray:
        filename = f"DEMAND/beneficiaries_within_cost_{cost}.tif" if not return_cost_window_difference else f"DEMAND/beneficiaries_within_cost_range_{cost}.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)

    def _get_proximity_raster_for_lu(self, lu) -> np.ndarray:
        filename = f"PROX/dr_{lu}.tif"
        return self._get_data_object(filename, nodata_replacement_value=0)
    
    def _get_minimum_cost_for_lu(self, lu) -> np.ndarray:
        filename = f'COSTS/minimum_cost_{lu}.tif'
        return self._get_data_object(filename)
    
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
        
        # set root path
        self.root_path = root_path                
        # make folders in root path
        self.make_environment()                 
        
        # get reference to the land use/land cover file
        lulc_file_path = self.get_file_path(land_use_filename, relative_to_root_path=True)
        self.land_use_map_reader = rasterio.open(lulc_file_path)

    def align_land_use_map(self, nodata_values: List[int], band: int = 1, reclassification_mappings: Dict[int, List[int]] = None):
        
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

    def reclassify(self, data_mtx, mappings: Dict[int, List[int]]) -> np.ndarray:

        self.printStepInfo("Raster reclassification")        
        current_task = self.get_task("[white]Reclassification", total=len(mappings.keys()))

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

    def detect_clumps(self, barrier_classes: List[int]) -> None:
        """Detect clumps as contiguous areas in the land-use raster that are separated by the specified barrier land-uses. Connectivity is defined as queens contiguity. 

        :param barrier_classes: Classes acting as barriers, i.e., separating clumps, defaults to [0]
        :type barrier_classes: List[int], optional
        """    
        self.printStepInfo("Detecting clumps")

        # barrier_classes are user-defined classes as well as nodata parts
        barrier_classes = barrier_classes + [self.nodata_value]

        lulc_data = self._get_land_use()
        barriers_mask = np.isin(lulc_data, barrier_classes, invert=False)
        lulc_data[barriers_mask] = 0

        clump_connectivity = np.full((3,3), 1)
        out_clumps = self._get_matrix(fill_value=0, shape=self._get_shape(), dtype=np.int32)

        nr_clumps = ndimage.label(lulc_data, structure=clump_connectivity, output=out_clumps)
        print(f"{Fore.YELLOW}{Style.BRIGHT} {nr_clumps} CLUMPS FOUND{Style.RESET_ALL}")
        
        # update clumps to hold nodata value where clump=0
        out_clumps[out_clumps == 0] = self.nodata_value
        self._write_file("BASE/clumps.tif", out_clumps, self._get_metadata(np.int32, self.nodata_value))        
        
        # done
        self.taskProgressReportStepCompleted()

    def mask_landuses(self, lu_classes: List[int] = None) -> None:
        """Generate land-use class masks (occurrence masks) for patch, edge, and built-up land-use classes.
        
        :param lu_classes: Classes for which to create class masks. If None, create class masks for all patch classes and edge classes, None by default.
        :type lu_classes: List[int], optional        
        """

        classes_for_masking = self.lu_classes_recreation_edge + self.lu_classes_recreation_patch if lu_classes is None else lu_classes

        # mask classes of interest into a binary raster to indicate presence/absence of recreational potential
        # we require this for all classes relevant to processing: patch and edge recreational classes, built-up classes
        self.printStepInfo("Creating land-use class masks")
        current_task = self.get_task('[white]Masking', total=len(classes_for_masking))

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
        """Detect edges (patch perimeters) of land-use classes that are defined as edge classes.

        :param lu_classes: List of classes for which edges should be assessed. If None, classes specified as classes.edge will be used, defaults to None
        :type lu_classes: List[int], optional
        :param ignore_edges_to_class: Class to which edges should be ignored, defaults to None
        :type ignore_edges_to_class: int, optional
        :param buffer_edges: Indicate classes for which edges should be buffered (expanded), defaults to None
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
            current_task = self.get_task("[white]Detecting edges", total=len(classes_to_assess))

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
    
    def sum_values_in_kernel(self, mtx_source: np.ndarray, mtx_clumps: np.ndarray, clump_slices: List[any], cost: float, mode: str, progress_task: any = None) -> np.ndarray:
        
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

    def class_total_supply(self, mode: str = 'ocv_filter2d') -> None:
        """Determines class total supply.

        :param mode: Method to perform sliding window operation. One of 'generic_filter', 'convolve', or 'ocv_filter2d'. Defaults to 'ocv_filter2d', defaults to 'ocv_filter2d'
        :type mode: str, optional
        """        

        dtype_to_use = np.int32

        # for each recreation patch class and edge class, determine total supply within cost windows
        # do this for each clump, i.e., operate only on parts of masks corresponding to clumps, ignore patches/edges external to each clump
        self.printStepInfo("Determining clumped supply per class")
        
        # clumps are required to properly mask islands
        rst_clumps, clump_nodata_mask = self._get_clumps()
        clump_slices = ndimage.find_objects(rst_clumps.astype(np.int64))
        
        step_count = len(clump_slices) * (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch)) * len(self.cost_thresholds)
        current_task = self.get_task("[white]Determining clumped supply", total=step_count)

        with self.progress:
            for c in self.cost_thresholds: 
                for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):    
                    
                    # determine source of list
                    lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
                    lu_mask = self._get_land_use_class_mask(lu)

                    # get result of windowed operation
                    lu_supply_mtx = self.sum_values_in_kernel(
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

        :param lu_weights: Dictionary of land-use class weights, where keys refer to land-use classes, and values to weights. If specified, weighted total supply will be determined, defaults to None
        :type lu_weights: Dict[any,float], optional
        :param write_non_weighted_result: Indicates if non-weighted total supply be computed, defaults to True
        :type write_non_weighted_result: bool, optional
        """        
                
        self.printStepInfo('Determining clumped total supply')

        # progress reporting        
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge))
        current_task = self.get_task("[white]Aggregating clumped supply", total=step_count)

        with self.progress if self._runsAsStandalone() else nullcontext() as bar:
            
            mtx_clumps, clump_nodata_mask = self._get_clumps()
                        
            for c in self.cost_thresholds:
                # get aggregation for current cost threshold
                current_total_supply_at_cost, current_weighted_total_supply_at_cost = self._get_aggregate_class_total_supply_for_cost(cost=c, lu_weights=lu_weights, write_non_weighted_result=write_non_weighted_result, task_progress=current_task)                                           

                # mask nodata areas and export total for costs
                if write_non_weighted_result:  
                    current_total_supply_at_cost[clump_nodata_mask] = self.nodata_value  
                    self._write_file(f"SUPPLY/totalsupply_cost_{c}.tif", current_total_supply_at_cost, self._get_metadata(np.float64, self.nodata_value))
                
                # export weighted total, if applicable
                if lu_weights is not None:                    
                    current_weighted_total_supply_at_cost[clump_nodata_mask] = self.nodata_value
                    self._write_file(f"SUPPLY/weighted_totalsupply_cost_{c}.tif", current_weighted_total_supply_at_cost, self._get_metadata(np.float64, self.nodata_value))

        # done
        self.taskProgressReportStepCompleted()
    
    def average_total_supply_across_cost(self, lu_weights: Dict[any, float] = None, cost_weights: Dict[float, float] = None, write_non_weighted_result: bool = True, write_scaled_result: bool = True) -> None:
        """Determine the total (recreational) land-use supply averaged across cost thresholds. Weighting of importance of land-uses and weighting of cost may be applied. 
           If either weighting schema (land-use classes or costs) is supplied, the total supply is determined as weighted average, i.e., the weighted sum of land-use class-specific supply, divided by the sum of weights.
           Potential combinations, i.e., land-use and subsequently cost-based weighting, are considered if both weighting schemas are supplied.

        :param lu_weights: Dictionary of land-use class weights, where keys refer to land-use classes, and values to weights. If specified, weighted total supply will be determined, defaults to None
        :type lu_weights: Dict[any, float], optional
        :param cost_weights: Dictionary of cost weights, where keys refer to cost thresholds, and values to weights. If specified, weighted total supply will be determined, defaults to None
        :type cost_weights: Dict[float, float], optional
        :param write_non_weighted_result: Indicates if non-weighted total supply be computed, defaults to True
        :type write_non_weighted_result: bool, optional
        :param write_scaled_result: Indicates if min-max-scaled values should be written as separate outputs, defaults to True
        :type write_scaled_result: bool, optional
        """        

        self.printStepInfo("Averaging supply across costs")
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge))
        current_task = self.get_task("[white]Averaging supply", total=step_count)

        # make result rasters
        # consider the following combinations

        # non-weighted lu + non-weighted cost (def. case)
        # non-weighted lu +     weighted cost (computed in addition to def. case if weights supplied)
        #     weighted lu + non-weighted cost (if weights applied only to previous step)
        #     weighted lu +     weighted cost

        with self.progress as p:

            # clumps are required to properly mask islands
            rst_clumps, clump_nodata_mask = self._get_clumps()

            # def. case
            if write_non_weighted_result:
                non_weighted_average_total_supply = self._get_matrix(0, self._get_shape(), np.float64)           
            # def. case + cost weighting
            if cost_weights is not None:
                cost_weighted_average_total_supply = self._get_matrix(0, self._get_shape(), np.float64)               

            if lu_weights is not None:
                # lu weights only
                lu_weighted_average_total_supply = self._get_matrix(0, self._get_shape(), np.float64)
                if cost_weights is not None:
                    # both weights
                    bi_weighted_average_total_supply = self._get_matrix(0, self._get_shape(), np.float64)

            # iterate over costs
            for c in self.cost_thresholds:

                # re-aggregate lu supply within cost, using currently supplied weights
                mtx_current_cost_total_supply, mtx_current_cost_weighted_total_supply = self._get_aggregate_class_total_supply_for_cost(c, lu_weights, write_non_weighted_result, current_task)                                           
                
                if write_non_weighted_result:
                    non_weighted_average_total_supply += mtx_current_cost_total_supply
                if cost_weights is not None:
                    cost_weighted_average_total_supply += (mtx_current_cost_total_supply * cost_weights[c])
                                            
                if lu_weights is not None:                                                            
                    lu_weighted_average_total_supply += mtx_current_cost_weighted_total_supply                    
                    if cost_weights is not None:                        
                        bi_weighted_average_total_supply += (mtx_current_cost_weighted_total_supply * cost_weights[c])
            
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

            if lu_weights is not None:
                # lu weights only
                lu_weighted_average_total_supply = lu_weighted_average_total_supply / len(self.cost_thresholds)
                lu_weighted_average_total_supply[clump_nodata_mask] = self.nodata_value
                self._write_file("INDICATORS/landuse_weighted_avg_totalsupply.tif", lu_weighted_average_total_supply, self._get_metadata(np.float64, self.nodata_value))
                
                # if write_scaled_result:
                #     # apply min-max scaling
                #     scaler = MinMaxScaler()
                #     lu_weighted_average_total_supply = scaler.fit_transform(lu_weighted_average_total_supply.reshape([-1,1]))
                #     self._write_dataset('INDICATORS/scaled_landuse_weighted_avg_totalsupply.tif', lu_weighted_average_total_supply.reshape(self.lsm_mtx.shape))

                if cost_weights is not None:
                    # both weights
                    bi_weighted_average_total_supply = bi_weighted_average_total_supply / sum(cost_weights.values())
                    bi_weighted_average_total_supply[clump_nodata_mask] = self.nodata_value
                    self._write_file("INDICATORS/bi_weighted_avg_totalsupply.tif", bi_weighted_average_total_supply, self._get_metadata(np.float64, self.nodata_value))

                    # if write_scaled_result:
                    #     # apply min-max scaling
                    #     scaler = MinMaxScaler()
                    #     bi_weighted_average_total_supply = scaler.fit_transform(bi_weighted_average_total_supply.reshape([-1,1]))
                    #     self._write_dataset('INDICATORS/scaled_bi_weighted_avg_totalsupply.tif', bi_weighted_average_total_supply.reshape(self.lsm_mtx.shape))

            
        # done
        self.taskProgressReportStepCompleted()

#endregion

    #
    # methods related to diversity indicators
    #
    #
    #
    
    def class_diversity(self) -> None:
        """Determine the diversity of land-use classes within cost thresholds. 
        """
        self.printStepInfo("Determining class diversity within costs")        

        step_count = (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch)) * len(self.cost_thresholds)
        current_task = self.get_task("[white]Determining class diversity", total=step_count)

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

        # done
        self.taskProgressReportStepCompleted()
    
    def average_diversity_across_cost(self, cost_weights: Dict[float, float] = None, write_non_weighted_result: bool = True, write_scaled_result: bool = True) -> None:
        """Determine diversity of (recreational) land-uses averaged across cost thresholds. 

        :param cost_weights: Dictionary of cost weights, where keys refer to cost thresholds, and values to weights. If specified, weighted total supply will be determined, defaults to None
        :type cost_weights: Dict[float, float], optional
        :param write_non_weighted_result: Indicates if non-weighted total supply be computed, defaults to True
        :type write_non_weighted_result: bool, optional
        :param write_scaled_result: Indicates if min-max-scaled values should be written as separate outputs, defaults to True
        :type write_scaled_result: bool, optional
        """        

        self.printStepInfo("Averaging diversity across costs")
        step_count = len(self.cost_thresholds)
        current_task = self.get_task("[white]Averaging diversity", total=step_count)

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
                
                mtx_current_diversity = self._get_diversity_for_cost(c) 
                mtx_current_diversity = mtx_current_diversity.astype(np.float64)
                
                if write_non_weighted_result:
                    average_diversity += mtx_current_diversity
                if cost_weights is not None:
                    cost_weighted_average_diversity += (average_diversity * cost_weights[c])
                
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

    
    #
    # methods related to the computation of demand
    #
    #
    #

    def disaggregation(self, population_grid: str, disaggregation_method: DisaggregationMethod, max_pixel_count: int, write_scaled_result: bool = True, count_threshold: int = None, min_sample_size: int = None) -> None:
        """Disaggregates population to specified built-up (residential) classes. 

        :param population_grid: Name of the population raster file to be used for disaggregation.
        :type population_grid: str
        :param disaggregation_method: Method to conduct disaggregation.
        :type disaggregation_method: DisaggregationMethod
        :param max_pixel_count: Number of built-up pixels per population raster. 
        :type max_pixel_count: int
        :param write_scaled_result: _description_, defaults to True
        :type write_scaled_result: bool, optional
        :param count_threshold: Sampling threshold.
        :type count_threshold: int, optional
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

        :param mode: Method to perform sliding window operation. One of 'generic_filter', 'convolve', or 'ocv_filter2d'. Defaults to 'ocv_filter2d', defaults to 'ocv_filter2d'
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
        current_task = self.get_task("[white]Determining beneficiaries", total=step_count)
        
        with self.progress:
            for c in self.cost_thresholds:

                mtx_pop_within_cost = self.sum_values_in_kernel(
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

        step_count = len(self.cost_thresholds)
        current_task = self.get_task("[white]Normalizing beneficiaries in cost ranges", total=step_count)
        with self.progress:
            # assert order from lowest to highest cost
            sorted_costs = sorted(self.cost_thresholds)
            self.progress.update(current_task, advance=1)
            
            # write lowest range directly
            mtx_lower_range = self._get_beneficiaries_for_cost(sorted_costs[0])
            mtx_lower_range[clump_nodata_mask] = self.nodata_value
            self._write_file(f"DEMAND/beneficiaries_within_cost_range_{sorted_costs[0]}.tif", mtx_lower_range, self._get_metadata(np.float64, self.nodata_value))
            del mtx_lower_range

            for i in range(1,len(sorted_costs)):
                
                mtx_lower_range = self._get_beneficiaries_for_cost(sorted_costs[i-1]) 
                mtx_current_cost = self._get_beneficiaries_for_cost(sorted_costs[i])

                mtx_beneficiaries_in_cost_range = self._get_matrix(0, self._get_shape(), np.float64)
                np.subtract(mtx_current_cost, mtx_lower_range, out=mtx_beneficiaries_in_cost_range)

                # mask nodata and export
                mtx_beneficiaries_in_cost_range[clump_nodata_mask] = self.nodata_value
                self._write_file(f"DEMAND/beneficiaries_within_cost_range_{sorted_costs[i]}.tif", mtx_beneficiaries_in_cost_range, self._get_metadata(np.float64, self.nodata_value))
                self.progress.update(current_task, advance=1)


        # done
        self.taskProgressReportStepCompleted()

    def average_beneficiaries_across_cost(self, cost_weights: Dict[float, float] = None, write_non_weighted_result: bool = True, write_scaled_result: bool = True) -> None:
        """Determine the number of potential beneficiaries, averaged across cost thresholds. 

        :param cost_weights: Dictionary of cost weights, where keys refer to cost thresholds, and values to weights. If specified, weighted total supply will be determined, defaults to None
        :type cost_weights: Dict[float, float], optional
        :param write_non_weighted_result: Indicates if non-weighted total supply be computed, defaults to True
        :type write_non_weighted_result: bool, optional
        :param write_scaled_result: Indicates if min-max-scaled values should be written as separate outputs, defaults to True
        :type write_scaled_result: bool, optional
        """        

        self.printStepInfo("Averaging beneficiaries across costs")
        step_count = len(self.cost_thresholds)
        current_task = self.get_task("[white]Averaging beneficiaries", total=step_count)

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
                mtx_current_pop = self._get_beneficiaries_for_cost(c, return_cost_window_difference=True)

                if write_non_weighted_result:
                    averaged_beneficiaries += mtx_current_pop
                if cost_weights is not None:
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


    #
    # methods related to the computation of cost-based indicators
    # 
    #
    #

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

    def compute_distance_rasters(self, mode: str = 'xr', lu_classes: List[int] = None, assess_builtup: bool = False) -> None:
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
        current_task = self.get_task("[white]Computing distance rasters", total=step_count)

        # iterate over classes and clumps
        with self.progress:
            for lu in classes_for_proximity_calculation:
                # make computation
                lu_dr = self._compute_proximity_raster_for_land_use(rst_clumps, clump_slices, lu, mode, current_task)
                # mask and export lu prox
                lu_dr[clump_nodata_mask] = self.nodata_value
                self._write_file(f"PROX/dr_{lu}.tif", lu_dr, self._get_metadata(np.float32, self.nodata_value))                
                # clean up
                del lu_dr
               
        if assess_builtup:
            step_count = len(self.lu_classes_builtup) * len(clump_slices)
            current_task = self.get_task("[white]Computing distance rasters to built-up", total=step_count)
            with self.progress:
                for lu in self.lu_classes_builtup:
                     # make computation
                    lu_dr = self._compute_proximity_raster_for_land_use(rst_clumps, clump_slices, lu, mode, current_task)
                    # mask and export lu prox
                    lu_dr[clump_nodata_mask] = self.nodata_value
                    self._write_file(f"PROX/dr_{lu}.tif", lu_dr, self._get_metadata(np.float32, self.nodata_value))                    
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
        current_task = self.get_task("[white]Assessing cost to closest", total=step_count)

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
        current_task = self.get_task("[white]Assessing minimum cost to closest", total=step_count)

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
        current_task = self.get_task("[white]Averaging cost to closest", total=step_count)

        # raster for average result
        mtx_average_cost = self._get_matrix(0, self._get_shape(), np.float32)
        mtx_lu_cost_count_considered = self._get_matrix(0, self._get_shape(), np.int32)    
        
        if distance_threshold > 0:
            print(f"{Fore.YELLOW}{Style.BRIGHT}Masking costs > {distance_threshold} units{Style.RESET_ALL}")                

        with self.progress as p:

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
        self._write_file('COSTS/cost_count.tif', mtx_lu_cost_count_considered, self._get_metadata(np.int32, self.nodata_value))
                
        np.divide(mtx_average_cost, mtx_lu_cost_count_considered, out=mtx_average_cost, where=mtx_lu_cost_count_considered > 0)     
        
        # we should now also be able to reset cells with nodata values to the actual nodata_value
        mtx_average_cost[mtx_lu_cost_count_considered < 0] = self.nodata_value

        self._write_file('INDICATORS/non_weighted_avg_cost.tif', mtx_average_cost, self._get_metadata(np.float32, self.nodata_value))
        
        # if write_scaled_result:
        #     # apply min-max scaling
        #     scaler = MinMaxScaler()
        #     mtx_average_cost = 1-scaler.fit_transform(mtx_average_cost.reshape([-1,1]))
        #     self._write_dataset('INDICATORS/scaled_non_weighted_avg_cost.tif', mtx_average_cost.reshape(self.lsm_mtx.shape), mask_nodata=False)

        del mtx_average_cost
        del mtx_lu_cost_count_considered

        # done
        self.taskProgressReportStepCompleted()


    #
    # methods related to the computation of flow-related indicators
    #
    #
    #

    def class_flow(self) -> None:
        """Determine the total number of potential beneficiaries (flow to given land-use classes) as the sum of total population, within cost thresholds.
        """

        self.printStepInfo("Determine class flow")        
        
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch))
        current_task = self.get_task("[white]Determine class-based flows within cost", step_count)

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
                    self._write_dataset(outfile_name, mtx_res, self._get_metadata(np.int32, self.nodata_value))
                    
                    del mtx_res
                    del mtx_lu

                    p.update(current_task, advance=1)

                del mtx_pop
        # done
        self.taskProgressReportStepCompleted()

    
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
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge)) 
        current_task = self.get_task("[white]Averaging flow across costs", total=step_count)

        with self.progress as p:

            # result grids for integrating averaged flows
            if write_non_weighted_result:
                integrated_average_flow = self._get_value_matrix()
            if cost_weights is not None:
                integrated_cost_weighted_average_flow = self._get_value_matrix()

            # iterate over cost thresholds and lu classes                
            for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):               

                # result grids for average flow for current cost threshold
                if write_non_weighted_result:
                    class_average_flow = self._get_value_matrix()
                if cost_weights is not None:
                    cost_weighted_class_average_flow = self._get_value_matrix()

                for c in self.cost_thresholds:
                    # determine source of list
                    lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
                    filename = "FLOWS/flow_class_{}_cost_{}.tif".format(lu, c) if lu_type == 'patch' else "FLOWS/flow_edge_class_{}_cost_{}.tif".format(lu, c)

                    mtx_current_flow = self._read_band(filename) 
                    if write_non_weighted_result:
                        class_average_flow += mtx_current_flow
                    if cost_weights is not None:
                        cost_weighted_class_average_flow += (mtx_current_flow * cost_weights[c])
                    p.update(current_task, advance=1)

                # we have now iterated over cost thresholds
                # export current class-averaged flow, and integrate with final product
                if write_non_weighted_result:
                    class_average_flow = class_average_flow / len(self.cost_thresholds)
                    self._write_dataset("FLOWS/average_flow_class_{}.tif".format(lu), class_average_flow)
                    # add to integrated grid
                    integrated_average_flow += class_average_flow

                if cost_weights is not None:
                    cost_weighted_class_average_flow = cost_weighted_class_average_flow / sum(cost_weights.values())
                    self._write_dataset("FLOWS/cost_weighted_average_flow_class_{}.tif".format(lu), cost_weighted_class_average_flow)
                    # add to integrated grid
                    integrated_cost_weighted_average_flow += cost_weighted_class_average_flow

            # export integrated grids
            if write_non_weighted_result:
                self._write_dataset("FLOWS/integrated_avg_flow.tif", integrated_average_flow)
                if write_scaled_result:
                    # apply min-max scaling
                    scaler = MinMaxScaler()
                    integrated_average_flow = scaler.fit_transform(integrated_average_flow.reshape([-1,1]))
                    self._write_dataset('FLOWS/scaled_integrated_avg_flow.tif', integrated_average_flow.reshape(self.lsm_mtx.shape))
            
            if cost_weights is not None:
                self._write_dataset("FLOWS/integrated_cost_weighted_avg_flow.tif", integrated_cost_weighted_average_flow)
                if write_scaled_result:
                    # apply min-max scaling
                    scaler = MinMaxScaler()
                    integrated_cost_weighted_average_flow = scaler.fit_transform(integrated_cost_weighted_average_flow.reshape([-1,1]))
                    self._write_dataset('FLOWS/scaled_integrated_cost_weighted_avg_flow.tif', integrated_cost_weighted_average_flow.reshape(self.lsm_mtx.shape))

        self.taskProgressReportStepCompleted()
