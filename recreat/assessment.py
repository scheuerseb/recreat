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

    # some status variables
    verbose_reporting = False

    # references to input data
    # this stores the lsm map as reference map
    lsm_rst = None
    lsm_mtx = None
    lsm_nodata_mask = None

    # distance units
    lsm_pixel_area_unit_factor = 1  # factor value to convert pixel area to km² through multiplication (1 px * factor = pixel area in km²) note: for CLC, this would be 0.01
    lsm_resolution = 1              # resolution of the land-use raster in km²

    # nodata value to use in replacements
    nodata_value = 0
    dtype = None

    # store params
    # define relevant recreation patch and edge classes, cost thresholds, etc.
    lu_classes_recreation_edge = []
    lu_classes_recreation_patch = []
    lu_classes_builtup = []
    cost_thresholds = []
    ignore_edges_to_classes = []
    
    # shared library
    clib = None 

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
        print("BYE BYE.")
        
    def make_environment(self) -> None:
        """Create required subfolders for raster files in the current scenario folder.
        """
        # create directories, if needed
        dirs_required = ['DEMAND', 'MASKS', 'SUPPLY', 'INDICATORS', 'TMP', 'FLOWS', 'CLUMPS_LU', 'PROX', 'COSTS', 'DIVERSITY']
        for d in dirs_required:
            current_path = f"{self.data_path}/{self.root_path}/{d}"
            if not os.path.exists(current_path):
                os.makedirs(current_path)


    def set_params(self, param_name: str, param_value: any) -> None:
        """Set processing parameters.

        :param paramName: Parameter, one of 'classes.edge', 'classes.patch', 'classes.builtup', 'costs', 'use-data-type', 'verbose-reporting'. 
        :type paramName: str
        :param paramValue: Parameter value, depending on parameter name.
        :type paramValue: any
        """        

        if param_name == 'classes.edge':
            self.lu_classes_recreation_edge = param_value
        elif param_name == 'classes.patch':
            self.lu_classes_recreation_patch = param_value
        elif param_name == 'classes.builtup':
            self.lu_classes_builtup = param_value
        elif param_name == 'costs':
            self.cost_thresholds = param_value    
        elif param_name == 'use-data-type':
            self.dtype = param_value 
        elif param_name == 'verbose-reporting':
            self.verbose_reporting = param_value
      
    def set_land_use_map(self, root_path: str, land_use_filename: str, nodata_values: list[float] = [0], nodata_fill_value: float = None) -> None:
        """Specify data sources for a given scenrio, i.e., root path, and import land-use raster file.

        :param root_path: Name of a scenario, i.e., subfolder within root of data path.
        :type root_path: str
        :param land_use_filename: Name of the land-use raster file for the given scenario.
        :type land_use_filename: str
        :param nodata_values: Values in the land-use raster that should be treated as nodata values, defaults to [0]
        :type nodata_values: list[float], optional
        :param nodata_fill_value: If set, specified nodata values in the land-use raster will be filled with the specified value, defaults to None.
        :type nodata_fill_value: float, optional
        """        

        self.root_path = root_path
                
        # check if folders are properly created in current scenario workspace
        self.make_environment()         
        
        # import lsm
        # support lazy-loading of data going forward
        # get only a reference to the raster, and require the data itself and the nodata mask only if needed.

        self.lsm_rst, self.lsm_mtx, self.lsm_nodata_mask = self._read_dataset(land_use_filename, nodata_values=nodata_values, nodata_fill_value = nodata_fill_value)

    #
    # The following classes will be called from asses_map_units. 
    # They will disaggregate population and determine clumped land-use class supplies.
    # Layers written will be specific to given costs.
    #
        
    def detect_clumps(self, barrier_classes: List[int] = [0]) -> None:
        """Detect clumps as contiguous areas in the land-use raster that are separated by the specified barrier land-uses. Connectivity is defined as queens contiguity. 

        :param barrier_classes: Classes acting as barriers, i.e., separating clumps, defaults to [0]
        :type barrier_classes: List[int], optional
        """        

        self.printStepInfo("Detecting clumps")
        clump_connectivity = np.full((3,3), 1)
        rst_clumps = self._get_value_matrix()

        indata = self.lsm_mtx.copy()
        barriers_mask = np.isin(indata, barrier_classes, invert=False)
        indata[barriers_mask] = 0

        nr_clumps = ndimage.label(indata, structure=clump_connectivity, output=rst_clumps)
        print(Fore.YELLOW + Style.BRIGHT + "{} CLUMPS FOUND".format(nr_clumps) + Style.RESET_ALL)
        self._write_dataset("MASKS/clumps.tif", rst_clumps)        
        
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
        self.printStepInfo("CREATING LAND-USE MASKS")
        current_task = self.get_task('[white]Masking land-uses', total=len(classes_for_masking))
        with self.progress:
            for lu in classes_for_masking:
                
                out_filename = self._get_file_path(f"MASKS/mask_{lu}.tif")
                if not os.path.isfile(out_filename):
                    current_lu_mask = self.lsm_mtx.copy()
                    # make mask for relevant pixels
                    mask = np.isin(current_lu_mask, [lu], invert=False)
                    # mask with binary values 
                    current_lu_mask[mask] = 1
                    current_lu_mask[~mask] = 0
                    self._write_dataset(f"MASKS/mask_{lu}.tif", current_lu_mask)

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
        if(len(classes_to_assess) > 0):
            
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

                    # read masking raster
                    mtx_mask = self._read_band("MASKS/mask_{}.tif".format(lu)) 
                    
                    # apply a 3x3 rectangular sliding window to determine pixel value diversity in window
                    rst_edgePixelDiversity = self._moving_window_generic(data_mtx=self.lsm_mtx, kernel_func=div_filter, kernel_size=3, kernel_shape='rect', dest_datatype=np.int16) 
                    rst_edgePixelDiversity = rst_edgePixelDiversity - 1
                    rst_edgePixelDiversity[rst_edgePixelDiversity > 1] = 1                
                    
                    # depending on whether to grow edge or not, intersect with land-use mask to have edge within land-use, or
                    # extending outside.
                    if not lu in buffer_edges:
                        mtx_mask = mtx_mask * rst_edgePixelDiversity
                        self._write_dataset("MASKS/edges_{}.tif".format(lu), mtx_mask)
                    else:
                        self._write_dataset("MASKS/edges_{}.tif".format(lu), rst_edgePixelDiversity)    

                    # some cleaning
                    del mtx_mask
                    del rst_edgePixelDiversity
        
                    self.progress.update(current_task, advance=1)

            # done
            self.taskProgressReportStepCompleted()

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
        rst_clumps = self._read_band("MASKS/clumps.tif")
        # detect slices here, to avoid having to recall detect_clumps each time we want to do proximity computations
        clump_slices = ndimage.find_objects(rst_clumps.astype(np.int64))        
        
        step_count = len(classes_for_proximity_calculation) * len(clump_slices)

        # if standalone, create new progress bar, otherwise use existing bar, and create task
        current_task = self.get_task("[white]Computing distance rasters", total=step_count)

        # iterate over classes and clumps
        with self.progress:
            for lu in classes_for_proximity_calculation:
                
                # target raster
                lu_dr = self._get_value_matrix()

                lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
                src_lu_mtx = self._read_band('MASKS/mask_{}.tif'.format(lu) if lu_type == "patch" else 'MASKS/edges_{}.tif'.format(lu))
                
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

                self._write_dataset("PROX/dr_{}.tif".format(lu), lu_dr)
                
                # clean up
                del lu_dr
                del src_lu_mtx


        if assess_builtup:
            
            step_count = len(clump_slices)
            current_task = self.get_task("[white]Computing distance rasters", total=step_count)
            with self.progress:
                
                # target raster
                lu_dr = self._get_value_matrix()
                # built-up source mask
                src_lu_mtx = self._read_band('MASKS/built-up.tif') 
                
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

                        # now all pixels outside of clump should be zeroed, and we can determine proximity on the subset of the full raster
                        sliced_dr = dr.DistanceRaster(sliced_lu_mtx, progress_bar=False)
                        sliced_dr = sliced_dr.dist_array
                        # proximities should only be written to clump object
                        sliced_dr[~obj_mask] = 0
                        lu_dr[obj_slice] += sliced_dr
                    
                    self.progress.update(current_task, advance=1)

                self._write_dataset("PROX/dr_built-up.tif".format(lu), lu_dr)
                del lu_dr
        
        # done
        self.taskProgressReportStepCompleted()

    def reclassify(self, mappings: Dict[int, List[int]], export_filename: str = None) -> None:
        """Reclassifies set(s) of source class values into a new destination class in the land-use dataset.

        :param mappings: Dictionary of new classes (keys), and corresponding list of class values to recategorize (values).
        :type mappings: Dict[int, List[int]]
        :param export_filename: Export to specified filename into root-path, defaults to None.
        :type export_filename: str  
        """        

        self.printStepInfo("Recategorizing classes")
        if self.lsm_mtx is not None:
            
            current_task = self.get_task("[white]Reclassification", total=len(mappings.keys()))

            # iterate over key-value combinations
            with self.progress:
                
                for (new_class_value, classes_to_aggregate) in mappings.items():
                    replacement_mask = np.isin(self.lsm_mtx, classes_to_aggregate, invert=False)
                    self.lsm_mtx[replacement_mask] = new_class_value
                    del replacement_mask
                    
                    self.progress.update(current_task, advance=1)                 

            # export to disk, if requested by user
            if export_filename is not None:
                print(f"{Fore.YELLOW}{Style.BRIGHT}Exporting reclassified land-use raster to {export_filename}{Style.RESET_ALL}")
                self._write_dataset(export_filename, self.lsm_mtx)

            # done
            self.taskProgressReportStepCompleted()

        else:
            print(Fore.WHITE + Back.RED + "ERR: Import Land-Use first" + Style.RESET_ALL)





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
                write_scaled_result=write_scaled_result
            )

            disaggregation_engine.run()

        
    def beneficiaries_within_cost(self, mode: str = 'ocv_filter2d') -> None:  
        """Determine number of beneficiaries within cost windows.

        :param mode: Method to perform sliding window operation. One of 'generic_filter', 'convolve', or 'ocv_filter2d'. Defaults to 'ocv_filter2d', defaults to 'ocv_filter2d'
        :type mode: str, optional
        """      
        self.printStepInfo("Determining beneficiaries within costs")

        mtx_disaggregated_population = self._read_band("DEMAND/disaggregated_population.tif")        
        mtx_clumps = self._read_band("MASKS/clumps.tif")
        clump_slices = ndimage.find_objects(mtx_clumps.astype(np.int64))        
        
        step_count = len(self.cost_thresholds) * len(clump_slices)
        current_task = self.get_task("[white]Determining beneficiaries", total=step_count)
        
        # this is actually constant per cost window
        infile_name = "DEMAND/disaggregated_population.tif" 
        
        with self.progress:
            for c in self.cost_thresholds:

                mtx_pop_within_cost = self.sum_values_in_kernel(
                    source_path=infile_name, 
                    mtx_clumps=mtx_clumps,
                    clump_slices=clump_slices,
                    cost=c,
                    mode=mode,
                    progress_task=current_task,
                    dest_datatype=np.float32
                )

                # export current beneficiaries within cost
                self._write_dataset(f"DEMAND/beneficiaries_within_cost_{c}.tif", mtx_pop_within_cost)
                del mtx_pop_within_cost
                
        # done
        self.taskProgressReportStepCompleted()


    def class_total_supply(self, mode: str = 'ocv_filter2d') -> None:
        """Determines class total supply.

        :param mode: Method to perform sliding window operation. One of 'generic_filter', 'convolve', or 'ocv_filter2d'. Defaults to 'ocv_filter2d', defaults to 'ocv_filter2d'
        :type mode: str, optional
        """        

        # for each recreation patch class and edge class, determine total supply within cost windows
        # do this for each clump, i.e., operate only on parts of masks corresponding to clumps, ignore patches/edges external to each clump
        self.printStepInfo("Determining clumped supply per class")
        # clumps are required to properly mask islands
        rst_clumps = self._read_band("MASKS/clumps.tif")
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
                        progress_task=current_task,
                        dest_datatype=np.int32
                    )

                    # export current cost
                    self._write_dataset(outfile_name, lu_supply_mtx)
                    del lu_supply_mtx           


        # done
        self.taskProgressReportStepCompleted()


    def sum_values_in_kernel(self, source_path: str, mtx_clumps: np.ndarray, clump_slices: List[any], cost: float, mode: str, progress_task: any = None, dest_datatype = np.int32) -> np.ndarray:
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
        :param dest_datatype: Datatype of target raster. By default, np.int32.
        :type dest_datatype: Numpy datatype.
        :return: Class supply for given land-use class within given cost.
        :rtype: np.ndarray
        """        

        # grid to store summed values in kernel 
        mtx_result = self._get_value_matrix(dest_datatype=dest_datatype)
        
        # get source raster for which values should be summe din kernel
        mtx_source = self._read_band(source_path)

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
            mtx_result[obj_slice] += sliding_supply.astype(dest_datatype)
            
            del sliding_supply
            del sliced_mtx_source

            if progress_task is not None:
                self.progress.update(progress_task, advance=1)
        
        # done with current iterations. return result
        del mtx_source
        return mtx_result
    


    


    #
    # 
    # 
    # 
    # The following functions are meant to be used / would be considered public methods.
    
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
            for c in self.cost_thresholds:
                # get aggregation for current cost threshold
                current_total_supply_at_cost, current_weighted_total_supply_at_cost = self._get_aggregate_class_total_supply_for_cost(cost=c, lu_weights=lu_weights, write_non_weighted_result=write_non_weighted_result, task_progress=current_task)                                           
                
                # export total for costs, if requested
                if write_non_weighted_result:                
                    self._write_dataset("INDICATORS/totalsupply_cost_{}.tif".format(c), current_total_supply_at_cost)                
                # export weighted total, if applicable
                if lu_weights is not None:                    
                    self._write_dataset("INDICATORS/weighted_totalsupply_cost_{}.tif".format(c), current_weighted_total_supply_at_cost)
                    
        # done
        self.taskProgressReportStepCompleted()

    


    def class_diversity(self) -> None:
        """Determine the diversity of land-use classes within cost thresholds. 
        """
        self.printStepInfo("Determining class diversity within costs")        
        
        step_count = (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch)) * len(self.cost_thresholds)        
        current_task = self.get_task("[white]Determining class diversity", total=step_count)

        with self.progress as p:
            for c in self.cost_thresholds:            
                mtx_diversity_at_cost = self._get_value_matrix(dest_datatype=np.int16)

                for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):                
                    # determine source of list
                    lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"                    
                    mtx_supply = self._get_supply_for_lu_and_cost(lu, lu_type, c)                   
                    mtx_supply[mtx_supply > 0] = 1
                    mtx_diversity_at_cost += mtx_supply
                    p.update(current_task, advance=1)
                
                # export current cost diversity
                self._write_dataset("DIVERSITY/diversity_cost_{}.tif".format(c), mtx_diversity_at_cost) 
                del mtx_diversity_at_cost

        # done
        self.taskProgressReportStepCompleted()


    #
    # Determine flow of beneficiaries to recreational opportunities per cost
    #

    def class_flow(self) -> None:
        """Determine the total number of potential beneficiaries (flow to given land-use classes) as the sum of total population, within cost thresholds.
        """
        self.printStepInfo("Determine class flow")        
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch))
        current_task = self.get_task("[white]Determine class-based flows within cost", step_count)

        with self.progress as p:            
            for c in self.cost_thresholds:
                mtx_pop = self._read_band("DEMAND/beneficiaries_within_cost_{}.tif".format(c))

                for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):                
                    
                    # determine source of list
                    lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
                    mtx_lu = self._get_mask_for_lu(lu, lu_type)
                    mtx_res = mtx_lu * mtx_pop
                    
                    # write result
                    outfile_name = "FLOWS/flow_class_{}_cost_{}.tif".format(lu, c) if lu_type == 'patch' else "FLOWS/flow_edge_class_{}_cost_{}.tif".format(lu, c)
                    self._write_dataset(outfile_name, mtx_res)
                    
                    del mtx_res
                    del mtx_lu

                    p.update(current_task, advance=1)

                del mtx_pop
        # done
        self.taskProgressReportStepCompleted()


    #
    #
    #
    # The following functions provide averaged metrics as targeted main indicators 

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

            # def. case
            if write_non_weighted_result:
                non_weighted_average_total_supply = self._get_value_matrix()            
            # def. case + cost weighting
            if cost_weights is not None:
                cost_weighted_average_total_supply = self._get_value_matrix()                

            if lu_weights is not None:
                # lu weights only
                lu_weighted_average_total_supply = self._get_value_matrix()
                if cost_weights is not None:
                    # both weights
                    bi_weighted_average_total_supply = self._get_value_matrix()

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
                self._write_dataset("INDICATORS/non_weighted_avg_totalsupply.tif", non_weighted_average_total_supply)
                if write_scaled_result:
                    # apply min-max scaling
                    scaler = MinMaxScaler()
                    non_weighted_average_total_supply = scaler.fit_transform(non_weighted_average_total_supply.reshape([-1,1]))
                    self._write_dataset('INDICATORS/scaled_non_weighted_avg_totalsupply.tif', non_weighted_average_total_supply.reshape(self.lsm_mtx.shape))


            # def. case + cost weighting
            if cost_weights is not None:
                cost_weighted_average_total_supply = cost_weighted_average_total_supply / sum(cost_weights.values())
                self._write_dataset("INDICATORS/cost_weighted_avg_totalsupply.tif", cost_weighted_average_total_supply)
                if write_scaled_result:
                    # apply min-max scaling
                    scaler = MinMaxScaler()
                    cost_weighted_average_total_supply = scaler.fit_transform(cost_weighted_average_total_supply.reshape([-1,1]))
                    self._write_dataset('INDICATORS/scaled_cost_weighted_avg_totalsupply.tif', cost_weighted_average_total_supply.reshape(self.lsm_mtx.shape))

            if lu_weights is not None:
                # lu weights only
                lu_weighted_average_total_supply = lu_weighted_average_total_supply / len(self.cost_thresholds)
                self._write_dataset("INDICATORS/landuse_weighted_avg_totalsupply.tif", lu_weighted_average_total_supply)
                if write_scaled_result:
                    # apply min-max scaling
                    scaler = MinMaxScaler()
                    lu_weighted_average_total_supply = scaler.fit_transform(lu_weighted_average_total_supply.reshape([-1,1]))
                    self._write_dataset('INDICATORS/scaled_landuse_weighted_avg_totalsupply.tif', lu_weighted_average_total_supply.reshape(self.lsm_mtx.shape))

                if cost_weights is not None:
                    # both weights
                    bi_weighted_average_total_supply = bi_weighted_average_total_supply / sum(cost_weights.values())
                    self._write_dataset("INDICATORS/bi_weighted_avg_totalsupply.tif", bi_weighted_average_total_supply)
                    if write_scaled_result:
                        # apply min-max scaling
                        scaler = MinMaxScaler()
                        bi_weighted_average_total_supply = scaler.fit_transform(bi_weighted_average_total_supply.reshape([-1,1]))
                        self._write_dataset('INDICATORS/scaled_bi_weighted_avg_totalsupply.tif', bi_weighted_average_total_supply.reshape(self.lsm_mtx.shape))

            
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

            # result raster
            if write_non_weighted_result:
                average_diversity = self._get_value_matrix()
            if cost_weights is not None:
                cost_weighted_average_diversity = self._get_value_matrix()

            # iterate over cost thresholds and aggregate cost-specific diversities into result
            for c in self.cost_thresholds:
                mtx_current_diversity = self._read_band("DIVERSITY/diversity_cost_{}.tif".format(c)) 
                if write_non_weighted_result:
                    average_diversity += mtx_current_diversity
                if cost_weights is not None:
                    cost_weighted_average_diversity += (average_diversity * cost_weights[c])

                p.update(current_task, advance=1)

            # export averaged diversity grids
            if write_non_weighted_result:
                average_diversity = average_diversity / len(self.cost_thresholds)
                self._write_dataset("INDICATORS/non_weighted_avg_diversity.tif", average_diversity)
                if write_scaled_result:
                    # apply min-max scaling
                    scaler = MinMaxScaler()
                    average_diversity = scaler.fit_transform(average_diversity.reshape([-1,1]))
                    self._write_dataset('INDICATORS/scaled_non_weighted_avg_diversity.tif', average_diversity.reshape(self.lsm_mtx.shape))
                        
            if cost_weights is not None:
                cost_weighted_average_diversity = cost_weighted_average_diversity / sum(cost_weights.values())
                self._write_dataset("INDICATORS/cost_weighted_avg_diversity.tif", cost_weighted_average_diversity)
                if write_scaled_result:
                    # apply min-max scaling
                    scaler = MinMaxScaler()
                    cost_weighted_average_diversity = scaler.fit_transform(cost_weighted_average_diversity.reshape([-1,1]))
                    self._write_dataset('INDICATORS/scaled_cost_weighted_avg_diversity.tif', cost_weighted_average_diversity.reshape(self.lsm_mtx.shape))

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

            # result raster
            if write_non_weighted_result:
                average_pop = self._get_value_matrix()
            if cost_weights is not None:
                cost_weighted_average_pop = self._get_value_matrix()

            # iterate over cost thresholds and aggregate cost-specific beneficiaries into result
            for c in self.cost_thresholds:
                mtx_current_pop = self._read_band("DEMAND/beneficiaries_within_cost_{}.tif".format(c)) 
                if write_non_weighted_result:
                    average_pop += mtx_current_pop
                if cost_weights is not None:
                    cost_weighted_average_pop += (mtx_current_pop * cost_weights[c])
                p.update(current_task, advance=1)
            
            # export averaged diversity grids
            if write_non_weighted_result:
                average_pop = average_pop / len(self.cost_thresholds)
                self._write_dataset("INDICATORS/non_weighted_avg_population.tif", average_pop)
                if write_scaled_result:
                    # apply min-max scaling
                    scaler = MinMaxScaler()
                    average_pop = scaler.fit_transform(average_pop.reshape([-1,1]))
                    self._write_dataset('INDICATORS/scaled_non_weighted_avg_population.tif', average_pop.reshape(self.lsm_mtx.shape))
            
            if cost_weights is not None:
                cost_weighted_average_pop = cost_weighted_average_pop / sum(cost_weights.values())
                self._write_dataset("INDICATORS/cost_weighted_avg_population.tif", cost_weighted_average_pop)
                if write_scaled_result:
                    # apply min-max scaling
                    scaler = MinMaxScaler()
                    cost_weighted_average_pop = scaler.fit_transform(cost_weighted_average_pop.reshape([-1,1]))
                    self._write_dataset('INDICATORS/scaled_cost_weighted_avg_population.tif', cost_weighted_average_pop.reshape(self.lsm_mtx.shape))

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


    # 
    # Clump detection in land-uses to determine size of patches and edges
    # To determine per-capita recreational area
    #

    def per_capita_opportunity_area(self):
        
        self.printStepInfo("Determining per-capita opportunity area")
        step_count = len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge)
        lu_progress = self.get_task("[white]Per-capita assessment", total=step_count)

        # get average flow for all lu patches as basis for average per-capita area
        mtx_average_flows = self._read_band("FLOWS/integrated_avg_flow.tif")
        # make clump raster
        clump_connectivity = np.full((3,3), 1)

        with self.progress as p:

            # particularly for debugging reasons, write out all grids
            res_lu_patch_area_per_capita = self._get_value_matrix()
            res_lu_clump_size = self._get_value_matrix()
            res_lu_clump_average_flow = self._get_value_matrix()

            # iterate over land-uses
            for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):
                # determine lu type
                lu_type = 'patch' if lu in self.lu_classes_recreation_patch else 'edge'
                
                # get lu mask as basis to determine clump total area
                mtx_current_lu_mask = self._get_mask_for_lu(lu, lu_type)

                mtx_current_lu_clumps = self._get_value_matrix()
                mtx_current_lu_clump_size = self._get_value_matrix()
                mtx_current_lu_clump_average_flow = self._get_value_matrix()

                nr_clumps = ndimage.label(mtx_current_lu_mask, structure=clump_connectivity, output=mtx_current_lu_clumps)
                print(Fore.YELLOW + Style.BRIGHT + "{} CLUMPS FOUND FOR CLASS {}".format(nr_clumps, lu) + Style.RESET_ALL)
                current_lu_clump_slices = ndimage.find_objects(mtx_current_lu_clumps.astype(np.int64)) 

                # iterate over clumps of current lu 
                clump_progress = p.add_task("[white]Iterate clumps on class {}".format(lu), total=len(current_lu_clump_slices))

                for patch_idx in range(len(current_lu_clump_slices)):
                    obj_slice = current_lu_clump_slices[patch_idx]
                    obj_label = patch_idx + 1

                    # get slice from mask
                    # mask
                    clump_slice = mtx_current_lu_clumps[obj_slice]
                    obj_mask = np.isin(clump_slice, [obj_label], invert=False)
                    
                    mask_slice = mtx_current_lu_mask[obj_slice].copy()
                    mask_slice[obj_mask] = 1
                    mask_slice[~obj_mask] = 0        
                    # now that we have zeroed all non-clump pixels, the area in sqkm of current clump should be equal to number of pixels of current clump
                    # convert from sqkm to sqm
                    val_clump_size = np.sum(mask_slice) * 1000000                   
                    
                    flow_slice = mtx_average_flows[obj_slice].copy()                    
                    flow_slice[~obj_mask] = 0
                    val_clump_average_flow = np.mean(flow_slice, where=flow_slice > 0) if np.sum(flow_slice) > 0 else 1

                    clump_size_as_mtx = val_clump_size * mask_slice
                    clump_flow_as_mtx = val_clump_average_flow * mask_slice
                    
                    mtx_current_lu_clump_size[obj_slice] += clump_size_as_mtx
                    mtx_current_lu_clump_average_flow[obj_slice] += clump_flow_as_mtx

                    
                    p.update(clump_progress, advance=1)
                
                # determine per-capita area
                mtx_current_lu_clump_area_per_capita = np.divide(mtx_current_lu_clump_size, mtx_current_lu_clump_average_flow, out=np.zeros_like(mtx_current_lu_clump_size), where=mtx_current_lu_clump_average_flow > 0)

                # export result
                #self._write_dataset('CLUMPS_LU/clump_size_class_{}.tif'.format(lu), mtx_current_lu_clump_size)
                #self._write_dataset('CLUMPS_LU/clump_flow_class_{}.tif'.format(lu), mtx_current_lu_clump_average_flow)
                #self._write_dataset('CLUMPS_LU/clump_pcap_class_{}.tif'.format(lu), mtx_current_lu_clump_area_per_capita)

                # add to integrated grid
                res_lu_clump_size += mtx_current_lu_clump_size
                res_lu_clump_average_flow += mtx_current_lu_clump_average_flow                
                res_lu_patch_area_per_capita += mtx_current_lu_clump_area_per_capita   

                p.update(lu_progress, advance=1)

        self._write_dataset('CLUMPS_LU/clumps_size.tif', res_lu_clump_size)
        self._write_dataset('CLUMPS_LU/clumps_flow.tif', res_lu_clump_average_flow)
        self._write_dataset('CLUMPS_LU/clumps_pcap.tif', res_lu_patch_area_per_capita)

        # done
        self.taskProgressReportStepCompleted()


    def minimum_cost_to_closest(self, write_scaled_result: bool = True) -> None:
        self.printStepCompleteInfo("Assessing minimum cost to closest")
        
        included_lu_classes = self.lu_classes_recreation_patch + self.lu_classes_recreation_edge
        step_count = len(included_lu_classes)
        current_task = self.get_task("[white]Assessing minimum cost to closest", total=step_count)

        # make result layer
        mtx_min_cost = self._get_value_matrix(fill_value=9999, dest_datatype=np.float32)

        with self.progress as p:
            for lu in included_lu_classes:

                mtx_proximity = self._read_band(f"PROX/dr_{lu}.tif")
                mtx_min_cost[mtx_proximity < mtx_]

        # done
        self.taskProgressReportStepCompleted()

    def cost_to_closest(self, lu_classes = None, nodata_value: int = -9999) -> None:
        
        self.printStepInfo("Assessing cost to closest")
        included_lu_classes = lu_classes if lu_classes is not None else self.lu_classes_recreation_patch + self.lu_classes_recreation_edge

        # we require clumps for masking
        mtx_clumps = self._read_band("MASKS/clumps.tif")        
        clump_slices = ndimage.find_objects(mtx_clumps.astype(np.int64))
        
        step_count = len(included_lu_classes) * len(clump_slices)
        current_task = self.get_task("[white]Assessing cost to closest", total=step_count)

        with self.progress as p:
            

            # iterate over land-uses
            for lu in included_lu_classes:
                
                # store final result
                mtx_out = self._get_value_matrix(dest_datatype=np.float32)
                
                # get relevant lu-specific datasets
                # complete cost raster
                mtx_lu_prox = self._read_band(f'PROX/dr_{lu}.tif')
                # complete mask raster
                lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
                mtx_lu_mask = self._get_mask_for_lu(lu, lu_type=lu_type)

                # iterate over patches, and for each patch, determine whether um of mask is 0 (then all 0 costs are nodata)
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
                        sliced_lu_prox = mtx_lu_prox[obj_slice].copy()
                        np.add(sliced_lu_prox, 1, out=sliced_lu_prox)                        
                        sliced_lu_prox[~obj_mask] = 0                      
                        mtx_out[obj_slice] += sliced_lu_prox

                    p.update(current_task, advance=1)

                # done iterating over patches

                del mtx_lu_mask
                del mtx_lu_prox


                # now apply nodata value to all values that are 0, as we shifted all proximities by +1
                mtx_out[mtx_out <= 0] = nodata_value
                np.subtract(mtx_out, 1, out=mtx_out, where=mtx_out > 0)

                # export mtx_out for current lu
                self._write_dataset(f'COSTS/minimum_cost_{lu}.tif', mtx_out)

        # done
        self.taskProgressReportStepCompleted()


    def average_cost_to_closest(self, lu_classes = None, distance_threshold: float = -1, out_of_distance_value: float = None, write_scaled_result: bool = True) -> None:

        # several assumptions need to be considered when computing costs:
        # the output of distances is...
        #   0 outside of clumps, as these are nodata areas (=nodata)
        # > 0 inside of clumps, when lu within clump (=proximity)
        #   0 inside of clumps, within lu of interest (=presence)
        #   0 inside of clumps, if lu not within clump (=nodata)  

        self.printStepInfo("Assessing average cost to closest")
        included_lu_classes = lu_classes if lu_classes is not None else self.lu_classes_recreation_patch + self.lu_classes_recreation_edge

        # we require clumps for masking
        mtx_clumps = self._read_band("MASKS/clumps.tif")        
        clump_slices = ndimage.find_objects(mtx_clumps.astype(np.int64))
        
        step_count = len(included_lu_classes) * len(clump_slices)
        current_task = self.get_task("[white]Assessing cost to closest", total=step_count)

        mask_value = self.nodata_value if out_of_distance_value is None else out_of_distance_value

        # raster for average result
        mtx_average_cost = self._get_value_matrix(dest_datatype=np.float32)
        mtx_lu_cost_count_considered = self._get_value_matrix(dest_datatype=np.float32)    
        
        with self.progress as p:

            # get built-up layer 
            if distance_threshold > 0:
                print(Fore.YELLOW + Style.BRIGHT + "APPLYING THRESHOLD MASKING" + Style.RESET_ALL)

            # now operate over clumps, in order to safe some computational time
            for lu in included_lu_classes:
                
                # get relevant lu-specific datasets
                # complete cost raster
                mtx_lu_prox = self._read_band(f'PROX/dr_{lu}.tif')
                # complete mask raster
                lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
                mtx_lu_mask = self._get_mask_for_lu(lu, lu_type=lu_type)

                # iterate over patches, and for each patch, determine whether um of mask is 0 (then all 0 costs are nodata)
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
                        sliced_lu_prox = mtx_lu_prox[obj_slice].copy()
                        np.add(sliced_lu_prox, 1, out=sliced_lu_prox)
                        
                        sliced_lu_prox[~obj_mask] = 0                      
                        mtx_average_cost[obj_slice] += sliced_lu_prox

                        # make mask of 1 to add to count mtx
                        sliced_lu_mask[obj_mask] = 1
                        mtx_lu_cost_count_considered[obj_slice] += sliced_lu_mask

                    else:
                        sliced_lu_mask[obj_mask] = 0                    
                        mtx_lu_cost_count_considered[obj_slice] += sliced_lu_mask
           

                    p.update(current_task, advance=1)

                del mtx_lu_mask
                del mtx_lu_prox


        # export average cost grid
        # prior, determine actual average. here, consider per each pixel the number of grids added.
        #mtx_average_cost[mtx_clumps <= 0] = -9999 
        self._write_dataset('COSTS/raw_sum_of_cost.tif', mtx_average_cost)
        self._write_dataset('COSTS/cost_count.tif', mtx_lu_cost_count_considered)
                
        # np.divide(mtx_average_cost, mtx_lu_cost_count_considered, out=mtx_average_cost, where=mtx_lu_cost_count_considered > 0)        
        # self._write_dataset('INDICATORS/non_weighted_avg_cost.tif', mtx_average_cost)
        
        # if write_scaled_result:
        #     # apply min-max scaling
        #     scaler = MinMaxScaler()
        #     mtx_average_cost = 1-scaler.fit_transform(mtx_average_cost.reshape([-1,1]))
        #     self._write_dataset('INDICATORS/scaled_non_weighted_avg_cost.tif', mtx_average_cost.reshape(self.lsm_mtx.shape))

        # del mtx_average_cost

        # done
        self.taskProgressReportStepCompleted()





    #
    # Helper functions
    #
    #
    def _get_dataset_reader(self, file_name: str, is_scenario_specific: bool = True) -> rasterio.DatasetReader:
        """Get dataset reader for a given raster file.

        :param file_name: Raster file for which a dataset reader should be returned.
        :type file_name: str
        :param is_scenario_specific: Indicates if the specified datasource located in a scenario-specific subfolder (True) or at the data path root (False), defaults to True
        :type is_scenario_specific: bool, optional
        :return: Dataset reader
        :rtype: rasterio.DatasetReader
        """
        path = self._get_file_path(file_name, is_scenario_specific)
        return rasterio.open(path)


    def _get_file_path(self, file_name: str, is_scenario_specific: bool = True):
        """Get the fully-qualified path to model file with specified filename.

        :param file_name: Model file for which the fully qualified path should be generated. 
        :type file_name: str
        :param is_scenario_specific: Indicates if the specified datasource located in a scenario-specific root-path (True) or at the data-path  (False), defaults to True.
        :type is_scenario_specific: bool, optional       
        """
        return (
            f"{self.data_path}/{file_name}"
            if not is_scenario_specific
            else f"{self.data_path}/{self.root_path}/{file_name}"
        )

    def _read_dataset(self, file_name: str, band: int = 1, nodata_values: List[float] = [0], is_scenario_specific: bool = True, nodata_fill_value = None, is_lazy_load = False) -> Tuple[rasterio.DatasetReader, np.ndarray, np.ndarray]:
        """Read a dataset and return reference to the dataset, values, and boolean mask of nodata values.

        :param file_name: Filename of dataset to be read.
        :type file_name: str
        :param band: Band to be read, defaults to 1
        :type band: int, optional
        :param nodata_values: List of values indicating nodata, defaults to [0]
        :type nodata_values: List[float], optional
        :param is_scenario_specific: Indicates if the specified datasource located in a scenario-specific subfolder (True) or at the data path root (False), defaults to True
        :type is_scenario_specific: bool, optional
        :param nodata_fill_value: If set to a value, nodata values of the raster to be imported will be filled up with the specified value, defaults to None
        :type nodata_fill_value: _type_, optional
        :param is_lazy_load: If set to True, apply lazy loading of raster data, defaults to False
        :type is_lazy_load: bool, optional
        :return: Dataset, data matrix, and mask of nodata values
        :rtype: Tuple[rasterio.DatasetReader, np.ndarray, np.ndarray]
        """        
                
        path = self._get_file_path(file_name, is_scenario_specific)
        if self.verbose_reporting:
            print(Fore.WHITE + Style.DIM + "    READING {}".format(path) + Style.RESET_ALL)
        
        rst_ref = rasterio.open(path)
        band_data = rst_ref.read(band)

        # attempt replacement of nodata with desired fill value
        nodata_mask = np.isin(band_data, nodata_values, invert=False)             
        
        if not is_lazy_load:

            fill_value = self.nodata_value if nodata_fill_value is None else nodata_fill_value
            for nodata_value in nodata_values:

                # replace only if not the same values!
                if fill_value != nodata_value:
                    if self.verbose_reporting:                
                        print(Fore.YELLOW + Style.DIM + "    REPLACING NODATA VALUE={} WITH FILL VALUE={}".format(nodata_value, fill_value) + Style.RESET_ALL) 
                    band_data = np.where(band_data==nodata_value, fill_value, band_data)

        else:
            del band_data
            band_data is None

        # determine nodata mask AFTER potential filling of nodata values  
        return rst_ref, band_data, nodata_mask


    def _read_band(self, file_name: str, band: int = 1, is_scenario_specific: bool = True) -> np.ndarray:
        """Read a raster band.

        :param file_name: Filename of dataset to be read.
        :type file_name: str
        :param band: Band to be read, defaults to 1
        :type band: int, optional
        :param is_scenario_specific: Indicates if the specified datasource located in a scenario-specific subfolder (True) or at the data path root (False), defaults to True
        :type is_scenario_specific: bool, optional
        :return: Raster band
        :rtype: np.ndarray
        """        

        rst_ref, band_data, nodata_mask = self._read_dataset(file_name=file_name, band=band, is_scenario_specific=is_scenario_specific, is_lazy_load=False)
        return band_data
    
    def _write_dataset(self, file_name: str, outdata: np.ndarray, mask_nodata: bool = True, is_scenario_specific: bool = True, custom_metadata: Dict[str,any] = None, custom_nodata_mask: np.ndarray = None) -> None:        
        """Write a dataset to disk.


        :param file_name: Name of file to be written.
        :type file_name: str
        :param outdata: Values to be written.
        :type outdata: np.ndarray
        :param mask_nodata: Indicates if nodata values should be masked using default or custom nodata mask (True) or not (False). Uses custom nodata mask if specified, defaults to True
        :type mask_nodata: bool, optional
        :param is_scenario_specific: Indicates whether file should be written in a scenario-specific subfolder (True) or in the data path root (False), defaults to True
        :type is_scenario_specific: bool, optional
        :param custom_metadata: Custom raster metadata to be used. If not specified, uses default land-use grid metadata, defaults to None
        :type custom_metadata: Dict[str,any], optional
        :param custom_nodata_mask: Custom nodata mask to apply if mask_nodata is set to True, defaults to None
        :type custom_nodata_mask: np.ndarray, optional
        """        

        custom_metadata = custom_metadata if custom_metadata is not None else self.lsm_rst.meta

        path = "{}/{}".format(self.data_path, file_name) if not is_scenario_specific else "{}/{}/{}".format(self.data_path, self.root_path, file_name)
        if self.verbose_reporting:
            print(Fore.WHITE + Style.DIM + "    WRITING {}".format(path) + Style.RESET_ALL)

        if mask_nodata is True:
            custom_nodata_mask = custom_nodata_mask if custom_nodata_mask is not None else self.lsm_nodata_mask
            outdata[custom_nodata_mask] = self.nodata_value    

        with rasterio.open(
            path,
            mode="w",
            driver="GTiff",
            height=outdata.shape[0],
            width=outdata.shape[1],
            count=1,
            dtype=outdata.dtype,
            crs=custom_metadata['crs'],
            transform=custom_metadata['transform']
        ) as new_dataset:
            new_dataset.write(outdata, 1)
    
    def _get_supply_for_lu_and_cost(self, lu, lu_type, cost):        
        # make filename
        filename = "SUPPLY/totalsupply_class_{}_cost_{}_clumped.tif".format(lu, cost) if lu_type == 'patch' else "SUPPLY/totalsupply_edge_class_{}_cost_{}_clumped.tif".format(lu, cost)
        # get supply of current class 
        lu_supply_mtx = self._read_band(filename) 
        # return supply
        return lu_supply_mtx
    
    def _get_mask_for_lu(self, lu, lu_type):        
        # make filename
        filename = "MASKS/mask_{}.tif".format(lu) if lu_type == 'patch' else "MASKS/edges_{}.tif".format(lu)
        # get mask of current class 
        lu_mask = self._read_band(filename) 
        # return mask
        return lu_mask

    def _get_value_matrix(self, fill_value: float = 0, shape: Tuple[int, int] = None, dest_datatype: any = None) -> np.ndarray:
        """Return array with specified fill value. 

        :param fill_value: Fill value, defaults to 0
        :type fill_value: float, optional
        :param shape: Shape of the matrix to be returned, defaults to None
        :type shape: Tuple[int, int], optional
        :param dest_datatype: Datatype of matrix, defaults to None
        :type dest_datatype: any, optional
        :return: Data matrix of given shape and fill value.
        :rtype: np.ndarray
        """        
        
        # determine parameters based on specified method arguments
        dest_dtype = self.dtype if dest_datatype is None else dest_datatype
        rst_dtype = self.lsm_mtx.dtype if dest_dtype is None else dest_dtype
        rst_shape = self.lsm_mtx.shape if shape is None else shape

        rst_new = np.full(shape=rst_shape, fill_value=fill_value, dtype=rst_dtype)
        return rst_new        

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

    def _kernel_sum(self, subarr: np.ndarray) -> float:
        """Determine the sum of values in a kernel window.

        Args:
            subarr (np.ndarray): Kernel.

        Returns:
            float: Sum of kernel values.
        """
        return(ndimage.sum(subarr))
    
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
        # define properties of result matrix
        # for the moment, use the dtype set by user
        if dest_datatype is None:
            dest_datatype = self.dtype

        target_dtype = self.lsm_mtx.dtype if dest_datatype is None else dest_datatype

        # make kernel
        kernel = self._get_circular_kernel(kernel_size) if kernel_shape == 'circular' else np.full((kernel_size, kernel_size), 1)
        # create result mtx as memmap
        mtx_res = np.memmap("{}/{}/TMP/{}".format(self.data_path, self.root_path, uuid.uuid1()), dtype=target_dtype, mode='w+', shape=data_mtx.shape) 
        # apply moving window over input mtx
        ndimage.generic_filter(data_mtx, kernel_func, footprint=kernel, output=mtx_res, mode='constant', cval=0)
        mtx_res.flush()
        return mtx_res
    
    def _moving_window_convolution(self, data_mtx: np.ndarray, kernel_size: int, kernel_shape: str = 'circular') -> np.ndarray: 

        # define properties of result matrix
        # for the moment, use the dtype set by user
        target_dtype = self.lsm_mtx.dtype if self.dtype is None else self.dtype
        # make kernel
        kernel = self._get_circular_kernel(kernel_size) if kernel_shape == 'circular' else np.full((kernel_size, kernel_size), 1)
        # create result mtx as memmap
        mtx_res = np.memmap("{}/{}/TMP/{}".format(self.data_path, self.root_path, uuid.uuid1()), dtype=target_dtype, mode='w+', shape=data_mtx.shape) 
        # apply convolution filter from ndimage that sums as weights are 0 or 1.        
        ndimage.convolve(data_mtx, kernel, output=mtx_res, mode = 'constant', cval = 0)        
        mtx_res.flush()
        return mtx_res
    
    def _moving_window_filter2d(self, data_mtx: np.ndarray, kernel_size: int, kernel_shape: str = 'circular') -> np.ndarray: 

        # define properties of result matrix
        # for the moment, use the dtype set by user
        target_dtype = self.lsm_mtx.dtype if self.dtype is None else self.dtype
        # make kernel
        radius = int(kernel_size / 2)
        kernel = self._get_circular_kernel(kernel_size) if kernel_shape == 'circular' else np.full((kernel_size, kernel_size), 1)
        # make sure that input is padded, as this determines border values
        data_mtx = np.pad(data_mtx, radius, mode='constant')
        mtx_res = cv.filter2D(data_mtx.astype(np.float32), -1, kernel)

        return mtx_res[radius:-radius,radius:-radius]
    
    def _get_aggregate_class_total_supply_for_cost(self, cost, lu_weights = None, write_non_weighted_result = True, task_progress = None):                        
        
        current_total_supply_at_cost = None
        current_weighted_total_supply_at_cost = None

        # make grids for the results: zero-valued grids with full lsm extent
        if write_non_weighted_result:
            current_total_supply_at_cost = self._get_value_matrix() 
        if lu_weights is not None:
            current_weighted_total_supply_at_cost = self._get_value_matrix()
        
        for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):                
            
            # determine source of list
            lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"                    
            lu_supply_mtx = self._get_supply_for_lu_and_cost(lu, lu_type, cost)

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
    
    

    


    

    def clean_temporary_files(self):
        """Clean temporary files from the TMP folder.
        """
        tmp_path = os.path.join(self.data_path, self.root_path, 'TMP')
        tmp_files = [os.path.join(self.data_path, self.root_path, 'TMP', f) for f in listdir(tmp_path) if isfile(join(tmp_path, f))]
        
        step_count = len(tmp_files)
        current_task = self.get_task('[red]Removing temporary files', total=step_count)
        
        for f in tmp_files:
            os.remove(f)
            self.progress.update(current_task, advance=1)
        
        self.taskProgressReportStepCompleted(msg = "TEMPORARY FILES CLEANED")
        
        
    
    
    

    
    







    

    


