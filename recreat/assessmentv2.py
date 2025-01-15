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

class Recreat2(RecreatBase):

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

    # reference to a dataset for the project
    lulc_dataset = None
   

    def __init__(self, data_path: str):        
        os.system('cls' if os.name == 'nt' else 'clear')

        if not os.path.exists(data_path) and os.access(data_path, os.W_OK):
            print(f"{Fore.RED}Error: data_path not found.{Style.RESET_ALL}")
            raise FileNotFoundError()

        else:
            super().__init__(data_path=data_path, root_path=None)
            print(Fore.WHITE + Style.BRIGHT + "recreat (C) 2024, Sebastian Scheuer" + Style.RESET_ALL)
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


    def set_land_use_map(self, root_path: str, land_use_filename: str) -> None:
        # set root path
        self.root_path = root_path                
        # make folders in root path
        self.make_environment()                 
        # get reference to the land use/land cover file
        lulc_file_path = self.get_file_path(land_use_filename, relative_to_root_path=True)
        self.lulc_dataset = rasterio.open(lulc_file_path)

    
    def get_file(self, filename, nodata_values: List[any], band: int = 1, relative_to_root_path: bool = True) -> Tuple[rasterio.DatasetReader, np.ndarray]:
        
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
    

    def write_file(self, filename, outdata, out_metadata, relative_to_root_path: bool = True):
        path = self.get_file_path(filename, relative_to_root_path)
        RecreatBase.write_output(path, outdata, out_metadata)


    def get_matrix(self, fill_value: float, shape: Tuple[int,int], dtype: any) -> np.ndarray:
        out_rst = np.full(shape=shape, fill_value=fill_value, dtype=dtype)
        return out_rst       

    def get_shape(self) -> Tuple[int,int]:
        return self.lulc_dataset.shape
    
    def get_metadata(self, new_dtype, new_nodata_value):
        out_meta = self.lulc_dataset.meta.copy()
        out_meta.update({
            'nodata' : new_nodata_value,
            'dtype' : new_dtype
        })
        return out_meta        




    def align_land_use_map(self, nodata_values: List[int], band: int = 1, reclassification_mappings: Dict[int, List[int]] = None):
        
        self.printStepInfo("Aligning land-use map")
        # conduct this if there is no base lulc file
        lulc_file_path = self.get_file_path("BASE/lulc.tif")
        if not os.path.isfile(lulc_file_path):
            # get lulc data from datasetreader 
            lulc_data = self.lulc_dataset.read(band)

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
            self.write_file("BASE/lulc.tif", lulc_data.astype(np.int32), self.get_metadata(np.int32, self.nodata_value))

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






    def detect_clumps(self, barrier_classes: List[int]) -> None:

        self.printStepInfo("Detecting clumps")

        # barrier_classes are user-defined classes as well as nodata parts
        barrier_classes = barrier_classes + [self.nodata_value]

        lulc_reader, lulc_data = self.get_file("BASE/lulc.tif", [self.nodata_value]) 
        barriers_mask = np.isin(lulc_data, barrier_classes, invert=False)
        lulc_data[barriers_mask] = 0

        clump_connectivity = np.full((3,3), 1)
        out_clumps = self.get_matrix(fill_value=0, shape=self.get_shape(), dtype=np.int32)

        nr_clumps = ndimage.label(lulc_data, structure=clump_connectivity, output=out_clumps)
        print(f"{Fore.YELLOW}{Style.BRIGHT} {nr_clumps} CLUMPS FOUND{Style.RESET_ALL}")
        
        # update clumps to hold nodata value where clump=0
        out_clumps[out_clumps == 0] = self.nodata_value
        self.write_file("BASE/clumps.tif", out_clumps, self.get_metadata(np.int32, self.nodata_value))        
        
        # done
        self.taskProgressReportStepCompleted()

    
    def mask_landuses(self, lu_classes: List[int] = None) -> None:

        classes_for_masking = self.lu_classes_recreation_edge + self.lu_classes_recreation_patch if lu_classes is None else lu_classes

        # mask classes of interest into a binary raster to indicate presence/absence of recreational potential
        # we require this for all classes relevant to processing: patch and edge recreational classes, built-up classes
        self.printStepInfo("Creating land-use class masks")
        current_task = self.get_task('[white]Masking', total=len(classes_for_masking))

        with self.progress:
            
            # import land-use dataset
            lulc_reader, lulc_data = self.get_file("BASE/lulc.tif", [self.nodata_value])
            # import clump dataset for masking of outputs
            clump_reader, clump_data = self.get_file("BASE/clumps.tif", [self.nodata_value])
            clump_nodata_mask = np.isin(clump_data, [self.nodata_value], invert=False)
            
            for lu in classes_for_masking:
                
                current_lu_mask = self.get_matrix(0, self.get_shape(), np.int32)

                # make mask for relevant pixels
                mask = np.isin(lulc_data, [lu], invert=False)                
                # mask with binary values 
                current_lu_mask[mask] = 1
                # mask with clump nodata
                current_lu_mask[clump_nodata_mask] = self.nodata_value
                
                # write to disk
                self.write_file(f"MASKS/mask_{lu}.tif", current_lu_mask, self.get_metadata(np.int32, self.nodata_value))

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
                lulc_reader, lulc_data = self.get_file("BASE/lulc.tif", [self.nodata_value]) 
                
                # import clump dataset for masking of outputs
                clump_reader, clump_data = self.get_file("BASE/clumps.tif", [self.nodata_value])
                clump_nodata_mask = np.isin(clump_data, [self.nodata_value], invert=False)
                
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
                        self.write_file(f"MASKS/edges_{lu}.tif", rst_edgePixelDiversity.astype(np.int32), self.get_metadata(np.int32, self.nodata_value))                        
                    
                    else:
                        # read masking raster, reconstruct original data by replacing nodata values with 0
                        mask_reader, mask_data = self.get_file(f"MASKS/mask_{lu}.tif", [self.nodata_value]) 
                        mask_data[clump_nodata_mask] = 0
                        mask_data = mask_data * rst_edgePixelDiversity
                        mask_data[clump_nodata_mask] = self.nodata_value

                        self.write_file(f"MASKS/edges_{lu}.tif", mask_data.astype(np.int32), self.get_metadata(np.int32, self.nodata_value))
                        del mask_data

                    # some cleaning
                    del rst_edgePixelDiversity
                    self.progress.update(current_task, advance=1)

            # done
            self.taskProgressReportStepCompleted()

    def _get_circular_kernel(self, kernel_size: int) -> np.ndarray:
        """Generate a kernel for floating-window operations with circular kernel mask.

        Args:
            kernel_size (int): Kernel diameter (in pixel). 

        Returns:
            np.ndarray: Circular kernel.
        """
        kernel = np.zeros((kernel_size,kernel_size))
        radius = kernel_size/2
        # modern scikit uses a tuple for center
        rr, cc = disk( (kernel_size//2, kernel_size//2), radius)
        kernel[rr,cc] = 1
        return kernel
    
    def _moving_window_generic(self, data_mtx: np.ndarray, kernel_func: Callable[[np.ndarray], float], kernel_size: int, kernel_shape: str = 'circular', dest_datatype = None) -> np.ndarray:
        """Conduct a moving window operation with specified kernel shape and kernel size on an array.

        Args:
            data_mtx (np.ndarray): Input array
            kernel_func (Callable[[np.ndarray], float]): Callable for aggregation/Summarization of values in kernel window.
            kernel_size (int): Size of kernel (total with for squared kernel window, kernel diameter for circular kernel window).
            kernel_shape (str, optional): Kernel shape: Circular kernel (circular) or squared/rectangular kernel (rect). Defaults to 'circular'.
            dest_datatype (any, optional): Destination datatype.

        Returns:
            np.ndarray: Output array
        """

        # make kernel
        kernel = self._get_circular_kernel(kernel_size) if kernel_shape == 'circular' else np.full((kernel_size, kernel_size), 1)
        mtx_res = self.get_matrix(0, self.get_shape(), np.int32)
        
        # apply moving window over input mtx
        ndimage.generic_filter(data_mtx, kernel_func, footprint=kernel, output=mtx_res, mode='constant', cval = 0)
        return mtx_res
    
    def _moving_window_convolution(self, data_mtx: np.ndarray, kernel_size: int, kernel_shape: str = 'circular') -> np.ndarray: 

        # make kernel
        kernel = self._get_circular_kernel(kernel_size) if kernel_shape == 'circular' else np.full((kernel_size, kernel_size), 1)
        # make result matrix
        mtx_res = self.get_matrix(0, self.get_shape(), np.int32)
        # apply convolution filter from ndimage that sums as weights are 0 or 1.        
        ndimage.convolve(data_mtx, kernel, output=mtx_res, mode = 'constant', cval = 0)        
        return mtx_res
    
    def _moving_window_filter2d(self, data_mtx: np.ndarray, kernel_size: int, kernel_shape: str = 'circular') -> np.ndarray: 
        
        # make kernel
        radius = kernel_size // 2
        kernel = self._get_circular_kernel(kernel_size) if kernel_shape == 'circular' else np.full((kernel_size, kernel_size), 1)
        # make sure that input is padded, as this determines border values
        data_mtx = np.pad(data_mtx, radius, mode='constant')
        data_mtx = data_mtx.astype(np.float64)        
        mtx_res = cv.filter2D(data_mtx, -1, kernel) 
        return mtx_res[radius:-radius,radius:-radius]
    




    def class_total_supply(self, mode: str = 'ocv_filter2d') -> None:
        """Determines class total supply.

        :param mode: Method to perform sliding window operation. One of 'generic_filter', 'convolve', or 'ocv_filter2d'. Defaults to 'ocv_filter2d', defaults to 'ocv_filter2d'
        :type mode: str, optional
        """        

        # for each recreation patch class and edge class, determine total supply within cost windows
        # do this for each clump, i.e., operate only on parts of masks corresponding to clumps, ignore patches/edges external to each clump
        self.printStepInfo("Determining clumped supply per class")
        
        # clumps are required to properly mask islands
        clump_reader, rst_clumps = self.get_file("BASE/clumps.tif", [self.nodata_value])
        clump_nodata_mask = np.isin(rst_clumps, [self.nodata_value], invert=False)
        clump_slices = ndimage.find_objects(rst_clumps.astype(np.int64))
        
        step_count = len(clump_slices) * (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch)) * len(self.cost_thresholds)
        current_task = self.get_task("[white]Determining clumped supply", total=step_count)

        with self.progress:
            for c in self.cost_thresholds: 
                for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):    
                    
                    # determine source of list
                    lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"

                    infile_name = (
                        f"MASKS/mask_{lu}.tif"
                        if lu_type == "patch"
                        else f"MASKS/edges_{lu}.tif"
                    )
                    outfile_name = (
                        f"SUPPLY/totalsupply_class_{lu}_cost_{c}_clumped.tif"
                        if lu_type == "patch"
                        else f"SUPPLY/totalsupply_edge_class_{lu}_cost_{c}_clumped.tif"
                    )


                    # get result of windowed operation
                    lu_supply_mtx = self.sum_values_in_kernel(
                        source_path=infile_name, 
                        mtx_clumps=rst_clumps, 
                        clump_slices=clump_slices, 
                        cost=c, 
                        mode=mode, 
                        progress_task=current_task
                    )

                    # mask nodata regions based on clumps
                    lu_supply_mtx[clump_nodata_mask] = self.nodata_value

                    # export current cost
                    self.write_file(outfile_name, lu_supply_mtx.astype(np.int32), self.get_metadata(np.int32, self.nodata_value))
                    del lu_supply_mtx           


        # done
        self.taskProgressReportStepCompleted()
    
    def sum_values_in_kernel(self, source_path: str, mtx_clumps: np.ndarray, clump_slices: List[any], cost: float, mode: str, progress_task: any = None) -> np.ndarray:
        """Compute total (sum) of values in source raster within a given cost.

        :param source_path: Path to source raster.
        :type source_path: str
        :param mtx_clumps: Array of clumps.
        :type mtx_clumps: np.ndarray
        :param clump_slices: List of clump slices.
        :type clump_slices: List[any]
        :param cost: Cost threshold.
        :type cost: float
        :param mode: Method to use to determine supply within cost. One of 'convolve', 'generic_filter', or 'ocv_filter2d'.
        :type mode: str
        :param progress_task: Progress task, defaults to None
        :type progress_task: any, optional      
        :return: Class supply for given land-use class within given cost.
        :rtype: np.ndarray
        """        

        # grid to store summed values in kernel 
        mtx_result = self.get_matrix(0, self.get_shape(), np.int32)
        
        # get source raster for which values should be summed in kernel
        # reset nodata values to 0 values for proper summation
        mtx_reader, mtx_source = self.get_file(source_path, [self.nodata_value])
        mtx_source = mtx_source.astype(np.int32)
        mtx_source[mtx_source == self.nodata_value] = 0

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
                sliding_supply = self._moving_window_convolution(sliced_mtx_source, cost)
            elif mode == 'generic_filter':                 
                sliding_supply = self._moving_window_generic(sliced_mtx_source, sum_filter, cost)
            elif mode == 'ocv_filter2d':
                sliding_supply = self._moving_window_filter2d(sliced_mtx_source, cost)
           
            sliding_supply[~obj_mask] = 0
            mtx_result[obj_slice] += sliding_supply.astype(np.int32)
            
            del sliding_supply
            del sliced_mtx_source

            if progress_task is not None:
                self.progress.update(progress_task, advance=1)
        
        # done with current iterations. return result
        del mtx_source
        return mtx_result