###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################


import os
from os import listdir
from os.path import isfile, join
import ctypes
import platform

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
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
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
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





class recreat:

    # some status variables
    verbose_reporting = False

    # environment variables
    data_path = None            # path to datasets
    root_path = "current"       # path to a specific "scenario" to be assessed, i.e., subfolder in data_path

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
    
    # progress reporting
    progress = None     
    task_assess_map_units = None

    # shared library
    clib = None 

    def __init__(self, data_path: str):
        os.system('cls' if os.name == 'nt' else 'clear')
        self.data_path = data_path  
        print(Fore.WHITE + Style.BRIGHT + "ReCreat (C) 2024, Sebastian Scheuer" + Style.RESET_ALL)
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
        dirs_required = ['DEMAND', 'MASKS', 'SUPPLY', 'INDICATORS', 'TMP', 'FLOWS', 'CLUMPS_LU', 'PROX', 'COSTS']
        for d in dirs_required:
            cpath = "{}/{}/{}".format(self.data_path, self.root_path, d)
            if not os.path.exists(cpath):
                os.makedirs(cpath)

    def printStepInfo(self, msg):
        print(Fore.CYAN + Style.BRIGHT + msg.upper() + Style.RESET_ALL)
    
    def printStepCompleteInfo(self, msg = "COMPLETED"):
        print(Fore.GREEN + Style.BRIGHT + msg + Style.RESET_ALL)

    def _new_progress(self, task_description, total):
        self.progress = self.get_progress_bar()
        task_new = self.progress.add_task(task_description, total=total)
        return task_new

    def get_progress_bar(self):
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn()
        )

    def set_params(self, paramName: str, paramValue: any) -> None:
        """ Set processing parameters.

        Args:
            paramType (str): Parameter, one of 'classes.edge', 'classes.patch', 'classes.builtup', 'costs', 'use-data-type', 'verbose-reporting'. 
            paramValue (any): Parameter value, depending on parameter name.
        """
        if paramName == 'classes.edge':
            self.lu_classes_recreation_edge = paramValue
        elif paramName == 'classes.patch':
            self.lu_classes_recreation_patch = paramValue
        elif paramName == 'classes.builtup':
            self.lu_classes_builtup = paramValue
        elif paramName == 'costs':
            self.cost_thresholds = paramValue    
        elif paramName == 'use-data-type':
            self.dtype = paramValue 
        elif paramName == 'verbose-reporting':
            self.verbose_reporting = paramValue
      
    def set_land_use_map(self, root_path: str, land_use_file: str, nodata_values: list[float] = [0], nodata_fill_value: float = None) -> None:
        """Specify data sources for a given scenrio, i.e., root path, and import land-use raster file.

        Args:
            root_path (str): Name of a scenario, i.e., subfolder within root of data path.
            land_use_file (str): Name of the land-use raster file for the given scenario.
            nodata_values (List[float], optional): Values in the land-use raster that should be treated as nodata values.  
            nodata_fill_value (float, optional): If set, specified nodata values in the land-use raster will be filled with the specified value.           
        """
        self.root_path = root_path
                
        # check if folders are properly created in current scenario workspace
        self.make_environment()         
        
        # import lsm
        self.lsm_rst, self.lsm_mtx, self.lsm_nodata_mask = self._read_dataset(land_use_file, nodata_values=nodata_values, nodata_fill_value = nodata_fill_value)


    #
    # The following classes will be called from asses_map_units. 
    # They will disaggregate population and determine clumped land-use class supplies.
    # Layers written will be specific to given costs.
    #
        
    def detect_clumps(self, barrier_classes: List[int] = [0]) -> None:
        """ Detect clumps as contiguous areas in the land-use raster that are separated by the specified barrier land-uses. Connectivity is defined as queens contiguity. 

        Args:
            barrier_classes (List[int], optional): _description_. Defaults to [0].
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

    def mask_landuses(self) -> None:
        """Generate land-use class masks (occurrence masks) for patch, edge, and built-up land-use classes.
        """
        # mask classes of interest into a binary raster to indicate presence/absence of recreational potential
        # we require this for all classes relevant to processing: patch and edge recreational classes, built-up classes
        self.printStepInfo("CREATING LAND-USE MASKS")
        classes_for_masking = self.lu_classes_recreation_edge + self.lu_classes_recreation_patch
        
        current_task = self._get_task('[white]Masking land-uses', total=len(classes_for_masking))
        
        with self.progress if self._runsAsStandalone() else nullcontext() as bar:        
            for lu in classes_for_masking:
                current_lu_mask = self.lsm_mtx.copy()
                # make mask for relevant pixels
                mask = np.isin(current_lu_mask, [lu], invert=False)
                # mask with binary values 
                current_lu_mask[mask] = 1
                current_lu_mask[~mask] = 0
                self._write_dataset("MASKS/mask_{}.tif".format(lu), current_lu_mask)
                self.progress.update(current_task, advance=1)

        # done    
        self.taskProgressReportStepCompleted()
    
    def detect_edges(self, lu_classes: List[int] = None, ignore_edges_to_class: int = None, grow_edges: bool = False, grow_factor: int = 1) -> None:
        """ Detect edges (patch perimeters) of land-use classes that are defined as edge classes.

        Args:
            lu_classes (List[int], optional): List of classes for which edges should be assessed. If None, classes specified as classes.edge will be used. Defaults to None. 
            ignore_edges_to_classes (int, optional): Class to which edges should be ignored. Defaults to None.
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
            current_task = self._get_task("[white]Detecting edges", total=len(classes_to_assess))
            
            with self.progress if self._runsAsStandalone() else nullcontext() as bar:
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
                    rst_edgePixelDiversity = self._moving_window_generic(self.lsm_mtx, div_filter, 3, 'rect') 
                    # test export at this step
                    self._write_dataset("MASKS/edges_{}_before_subtract.tif".format(lu), rst_edgePixelDiversity)


                    rst_edgePixelDiversity = rst_edgePixelDiversity - 1

                    rst_edgePixelDiversity[rst_edgePixelDiversity > 1] = 1                
                    
                    mtx_mask = mtx_mask * rst_edgePixelDiversity
                    self._write_dataset("MASKS/edges_{}.tif".format(lu), mtx_mask)
                    self.progress.update(current_task, advance=1)
        
            # done
            self.taskProgressReportStepCompleted()

    def compute_distance_rasters(self, mode: str = 'xr', lu_classes: List[int] = None, assess_builtup: bool = False) -> None:
        """Generate proximity rasters to land-use classes based on identified clumps.

        Args:
            mode (str, optional): Method used to compute proximity matrix. Either 'dr' or 'xr'. Defaults to 'xr'.
            lu_classes (List[int], optional): List of integers, i.e., land-use classes to assess. Defaults to None.
            assess_builtup (bool, optional): Assesses proximities to built-up, if true. Defaults to False.
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
        current_task = self._get_task("[white]Computing distance rasters", total=step_count)

        # iterate over classes and clumps
        with self.progress if self._runsAsStandalone() else nullcontext() as bar:
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

        if assess_builtup:
            
            step_count = len(clump_slices)
            current_task = self._get_task("[white]Computing distance rasters", total=step_count)
            with self.progress if self._runsAsStandalone() else nullcontext() as bar:
                
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
        
        # done
        self.taskProgressReportStepCompleted()

    def aggregate_classes(self, aggregations: Dict[int, List[int]]) -> None:
        """Replaces values of classes to aggregate with a new class value in the land-use dataset.

        Args:
            aggregations (Dict[int, List[int]]): Dictionary of intended classes (keys), and corresponding list of classes to aggregate (values). 
        """
        self.printStepInfo("Aggregating classes")
        if self.lsm_mtx is not None:
            
            current_task = self._get_task("[white]Aggregating classes", total=len(aggregations.keys()))

            # iterate over key-value combinations
            with self.progress if self._runsAsStandalone() else nullcontext() as bar:
                
                for (new_class_value, classes_to_aggregate) in aggregations.items():
                    replacement_mask = np.isin(self.lsm_mtx, classes_to_aggregate, invert=False)
                    self.lsm_mtx[replacement_mask] = new_class_value
                    del replacement_mask

                    self.progress.update(current_task, advance=1)                 

            # done
            self.printStepCompleteInfo()

        else:
            print(Fore.WHITE + Back.RED + "ERR: Import Land-Use first" + Style.RESET_ALL)


    def aggregate_builtup_classes(self) -> None:
        """ Aggregates specified built-up (i.e., potentially residential) classes into a single built-up layer. 
        """
        self.printStepInfo("Aggregating built-up land-use classes")
        current_task = self._get_task("[white]Aggregating built-up", total=1)
        with self.progress if self._runsAsStandalone() else nullcontext() as bar:
            
            # we don't add up independent classes anymore, but just grab all builtup classes, and make value replacements on the original input matrix 
            mtx_builtup = self.lsm_mtx.copy()                        
            builtup_mask = np.isin(mtx_builtup, self.lu_classes_builtup, invert=False)
            mtx_builtup[builtup_mask] = 1
            mtx_builtup[~builtup_mask] = 0         
                            
            # write built-up raster to disk
            self.progress.update(current_task, advance=1)                 
            self._write_dataset("MASKS/built-up.tif", mtx_builtup)
        
        # done
        self.printStepCompleteInfo()


    def _reproject_builtup_to_population(self, population_grid: str) -> None:
        """ Matches population and built-up rasters and sums built-up pixels within each cell of the population raster. It will write a raster of built-up pixel count.

        Args:
            population_grid (str): Name of the population raster file to be used for disaggregation.
        """
        self.printStepInfo('Reprojecting built-up')
        current_task = self._get_task("[white]Reprojection", total=1)
        with self.progress if self._runsAsStandalone() else nullcontext() as bar:
            
            # raster 1 = builtup
            src1, mtx_builtup, nodata_builtup = self._read_dataset("MASKS/built-up.tif")
            meta1 = src1.meta.copy()

            print(meta1)

            # raster 2 = pop
            src2, mtx_pop, nodata_pop = self._read_dataset(population_grid)
            meta2 = src2.meta.copy()

            print(meta2)

            transform, width, height = calculate_default_transform(src1.crs, meta2['crs'], meta2['width'], meta2['height'], *src2.bounds)                        
            meta1.update({
                'crs': meta2['crs'],
                'transform': transform,
                'width': width,
                'height': height
            })

            mtx_sum_of_builtup = np.zeros((height, width), dtype=rasterio.float32)
            
            reproject(
                source=mtx_builtup,
                destination=mtx_sum_of_builtup,
                src_transform=src1.transform,
                src_crs=src1.crs,
                dst_transform=transform,
                dst_crs=meta2['crs'],
                resampling=Resampling.sum
            )


            # export
            # this is the number of builtup pixels per pop raster grid cell
            self._write_dataset('DEMAND/builtup_count.tif', mtx_sum_of_builtup, src2)
            self.progress.update(current_task, advance=1)
        
        # done
        self.printStepCompleteInfo()

    def _conduct_disaggregation(self, population_grid: str) -> None:
        """ Applies disaggregation algorithm. At the moment, a simple area-weighted approach is implemented.

        Args:
            population_grid (str): Name of the population raster file to be used for disaggregation.
        """
        current_task = self._get_task("[white]Applying disaggregation algorithm", total=1)
        with self.progress if self._runsAsStandalone() else nullcontext() as bar:            
            # read the built-up patch count 
            ref_pop, mtx_pop, nodata_pop = self._read_dataset(population_grid)        
            ref_patch_count, mtx_patch_count, nodata_patch_count = self._read_dataset('DEMAND/builtup_count.tif')
            
            mtx_patch_population = self._get_value_matrix(shape=mtx_pop.shape).astype(np.float32)
            
            # make rasters the same data type
            # make sure that a float dtype is set          
            np.divide(mtx_pop.astype(np.float32), mtx_patch_count.astype(np.float32), out=mtx_patch_population, where=mtx_patch_count > 0)
            
            self._write_dataset('DEMAND/patch_population.tif', mtx_patch_population, custom_grid_reference=ref_pop, custom_nodata_mask=nodata_patch_count)
            self.progress.update(current_task, advance=1)     
            
            del(mtx_pop)
            del(mtx_patch_count)
            del(mtx_patch_population)

    def _reproject_patch_population_to_builtup(self) -> None:
        """ Matches patch population and built-up rasters. It will write the reprojected dataset to disk.
        """
        self.printStepInfo('Reprojecting built-up')
        current_task = self._get_task("[white]Reprojection", total=1)
        with self.progress if self._runsAsStandalone() else nullcontext() as bar:
            
            # raster 1 = builtup
            src1, mtx_patch_population, nodata_pop = self._read_dataset("DEMAND/patch_population.tif")
            meta1 = src1.meta.copy()
            
            # raster 2 = pop
            src2, mtx_builtup, nodata_builtup = self._read_dataset("MASKS/built-up.tif")
            meta2 = src2.meta.copy()

            transform, width, height = calculate_default_transform(src1.crs, meta2['crs'], meta2['width'], meta2['height'], *src2.bounds)                        
            meta1.update({
                'crs': meta2['crs'],
                'transform': transform,
                'width': width,
                'height': height
            })

            mtx_reprojected_patch_population = np.zeros((height, width), dtype=rasterio.float32)
            
            reproject(
                source=mtx_patch_population,
                destination=mtx_reprojected_patch_population,
                src_transform=src1.transform,
                src_crs=src1.crs,
                dst_transform=transform,
                dst_crs=meta2['crs'],
                resampling=Resampling.min
            )

            # export
            # this is the number of builtup pixels per pop raster grid cell
            self._write_dataset('DEMAND/reprojected_patch_population.tif', mtx_reprojected_patch_population, src2)
            self.progress.update(current_task, advance=1)
        
        # done
        self.printStepCompleteInfo()


    def disaggregate_population(self, population_grid: str, force_computing: bool = False, write_scaled_result: bool = True) -> None:
        """Aggregates built-up land-use classes into a single raster of built-up areas, and intersects built-up with the scenario-specific population grid to provide disaggregated population.
           The method currently implements a simple area-weighted disaggregation method. This method can currently account for gridded land-use and population featuring the same extent and resolution,
           and for the gridded population to have a lower resolution than gridded land-use 

        Args:
            population_grid (str): Name of the population raster file to be used for disaggregation.
            force_computing (bool, optional): Force (re-)computation of intermediate products if they already exist. 
            write_scaled_result (bool, optional): Export min-max scaled result, if True. Defaults to True.
        """
        self.printStepInfo("Disaggregating population to built-up")
        

        # cases to consider:
        # A -- pop and built-up (hence, land use) have the same resolution and extent
        # B -- pop has a lower resolution (and differing extent?) than built-up
        # TODO: C -- built-up has a lower resolution than pop 
        # TODO: Test if resolutions actually differ!

        # disaggregation in multiple steps
        # we require built-up to be available
        if not os.path.isfile("{}/{}/MASKS/built-up.tif".format(self.data_path, self.root_path)) or force_computing:
            self.aggregate_builtup_classes()
        else:
            print(Style.DIM + "    Skip aggregation of built-up classes. File exists." + Style.RESET_ALL)

        # first: Aggregate built-up pixels per population grid cell to determine patch count 
        if not os.path.isfile("{}/{}/DEMAND/builtup_count.tif".format(self.data_path, self.root_path)) or force_computing:
            self._reproject_builtup_to_population(population_grid=population_grid)
        else:
            print(Style.DIM + "    Skip reprojection of built-up. File exists." + Style.RESET_ALL)


        # second: Determine patch population
        if not os.path.isfile("{}/{}/DEMAND/patch_population.tif".format(self.data_path, self.root_path)) or force_computing:            
            self._conduct_disaggregation(population_grid=population_grid)
        else:
            print(Style.DIM + "    Skip estimation of patch population. File exists." + Style.RESET_ALL)

        # third: Reproject patch population to match built-up grid
        if not os.path.isfile("{}/{}/DEMAND/reprojected_patch_population.tif".format(self.data_path, self.root_path)) or force_computing:
            self._reproject_patch_population_to_builtup()
        else:
            print(Style.DIM + "    Skip reprojection of patch population. File exists." + Style.RESET_ALL)

                                               
        # fourth: intersect patch population with built-up patches
        if not os.path.isfile("{}/{}/DEMAND/disaggregated_population.tif".format(self.data_path, self.root_path)) or force_computing:            
            mtx_pop = self._read_band('DEMAND/reprojected_patch_population.tif')                             
            mtx_builtup = self._read_band('MASKS/built-up.tif')
            # intersect 
            mtx_builtup = mtx_builtup * mtx_pop        
            self._write_dataset("DEMAND/disaggregated_population.tif", mtx_builtup)
        
            if write_scaled_result:
                scaler = MinMaxScaler()
                mtx_builtup = scaler.fit_transform(mtx_builtup.reshape([-1,1]))
                self._write_dataset("DEMAND/scaled_disaggregated_population.tif", mtx_builtup.reshape(self.lsm_mtx.shape))
        else:
            print(Style.DIM + "    Skip disaggregation of patch population. File exists." + Style.RESET_ALL)

        # done   
        self.taskProgressReportStepCompleted()

        
    def beneficiaries_within_cost(self):        
        self.printStepInfo("Determining beneficiaries within costs")

        # get relevant input data
        mtx_disaggregated_population = self._read_band("DEMAND/disaggregated_population.tif")        
        # also beneficiaries need to be clumped
        rst_clumps = self._read_band("MASKS/clumps.tif")
        clump_slices = ndimage.find_objects(rst_clumps.astype(np.int64))        
        
        step_count = len(self.cost_thresholds) * len(clump_slices)
        current_task = self._get_task("[white]Determining beneficiaries", total=step_count)
        
        with self.progress if self._runsAsStandalone() else nullcontext() as bar:

            for c in self.cost_thresholds:

                mtx_pop_within_cost = self._get_value_matrix()

                # now operate over clumps, in order to safe some computational time
                for patch_idx in range(len(clump_slices)):
                    obj_slice = clump_slices[patch_idx]
                    obj_label = patch_idx + 1

                    # get slice from land-use mask
                    sliced_pop_mtx = mtx_disaggregated_population[obj_slice].copy() 
                    sliced_clump_mtx = rst_clumps[obj_slice]

                    # properly mask out current object
                    obj_mask = np.isin(sliced_clump_mtx, [obj_label], invert=False)
                    sliced_pop_mtx[~obj_mask] = 0

                    # now all pixels outside of clump should be zeroed, and we can determine total supply within sliding window
                    sliding_pop = self._moving_window_generic(sliced_pop_mtx, self._kernel_sum, c)
                    sliding_pop[~obj_mask] = 0
                    mtx_pop_within_cost[obj_slice] += sliding_pop
                    
                    del sliding_pop
                    del sliced_pop_mtx
                    # progress reporting
                    self.progress.update(current_task, advance=1)

                self._write_dataset("DEMAND/beneficiaries_within_cost_{}.tif".format(c), mtx_pop_within_cost)

        # done
        self.taskProgressReportStepCompleted()


    def class_total_supply(self, mode = 'ocv_filter2d'):
        """Determine class total supply.

        Args:
            mode (str, optional): Method to perform sliding window operation. One of 'generic_filter', 'convolve', or 'ocv_filter2d'. Defaults to 'ocv_filter2d'.
        """
        # for each recreation patch class and edge class, determine total supply within cost windows
        # do this for each clump, i.e., operate only on parts of masks corresponding to clumps, ignore patches/edges external to each clump
        self.printStepInfo("Determining clumped supply per class")
        # clumps are required to properly mask islands
        rst_clumps = self._read_band("MASKS/clumps.tif")
        clump_slices = ndimage.find_objects(rst_clumps.astype(np.int64))        

        step_count = len(clump_slices) * (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch)) * len(self.cost_thresholds)
        current_task = self._get_task("[white]Determining clumped supply", total=step_count)

        with self.progress if self._runsAsStandalone() else nullcontext() as bar:
            for c in self.cost_thresholds:                 
                for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):                
                    # determine source of list
                    lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"

                    infile_name = "MASKS/mask_{}.tif".format(lu) if lu_type == "patch" else "MASKS/edges_{}.tif".format(lu)
                    outfile_name = "SUPPLY/totalsupply_class_{}_cost_{}_clumped.tif".format(lu, c) if lu_type == "patch" else "SUPPLY/totalsupply_edge_class_{}_cost_{}_clumped.tif".format(lu, c)

                    # get result of windowed operation
                    lu_supply_mtx = self._class_total_supply_for_lu_and_cost(mask_path=infile_name, rst_clumps=rst_clumps, clump_slices=clump_slices, cost=c, mode=mode, progress_task=current_task)                            
                    # export current cost
                    self._write_dataset(outfile_name, lu_supply_mtx)                
                    

        # done
        self.taskProgressReportStepCompleted()


    def _class_total_supply_for_lu_and_cost(self, mask_path: str, rst_clumps: np.ndarray, clump_slices: List[any], cost: float, mode: str, progress_task: any = None) -> np.ndarray:
        """Compute supply of land-use within a given cost.

        Args:
            mask_path (str): Path to land-use mask.
            rst_clumps (np.ndarray): Array of clumps.
            clump_slices (List[any]): List of clump slices.
            cost (float): Cost threshold.
            mode (str): Method to use to determine supply within cost. One of 'convolve', 'generic_filter', or 'ocv_filter2d'.
            progress_task (any, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        # grid to store lu supply 
        lu_supply_mtx = self._get_value_matrix().astype(np.int32)
        # get land-use current mask
        full_lu_mtx = self._read_band(mask_path)

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
            sliced_lu_mtx = full_lu_mtx[obj_slice].copy() 
            sliced_clump_mtx = rst_clumps[obj_slice]

            # properly mask out current object
            obj_mask = np.isin(sliced_clump_mtx, [obj_label], invert=False)
            sliced_lu_mtx[~obj_mask] = 0

            # now all pixels outside of clump should be zeroed, and we can determine total supply within sliding window
            if mode == 'convolve':            
                sliding_supply = self._moving_window_convolution(sliced_lu_mtx, cost)
            elif mode == 'generic_filter':                 
                sliding_supply = self._moving_window_generic(sliced_lu_mtx, sum_filter, cost)
            elif mode == 'ocv_filter2d':
                sliding_supply = self._moving_window_filter2d(sliced_lu_mtx, cost)

           
            sliding_supply[~obj_mask] = 0
            lu_supply_mtx[obj_slice] += sliding_supply.astype(np.int32)
            
            del sliding_supply
            del sliced_lu_mtx

            if progress_task is not None:
                self.progress.update(progress_task, advance=1)
        
        # done with current iterations. return result
        del full_lu_mtx
        return lu_supply_mtx
    

    def _get_task(self, task_description, total):
        if self._runsAsStandalone():
            current_task = self._new_progress(task_description, total=total)
        else:
            current_task = self.progress.add_task(task_description, total=total)
        return current_task
    


    #
    # 
    # 
    # 
    # The following functions are meant to be used / would be considered public methods.
    
    def aggregate_class_total_supply(self, lu_weights: Dict[any,float] = None, write_non_weighted_result: bool = True) -> None:
        """Aggregate total supply of land-use classes within each specified cost threshold. A weighting schema may be supplied, in which case a weighted average is determined as the sum of weighted class supply divided by the sum of all weights.

        Args:
            lu_weights (Dict[any,float], optional): Dictionary of land-use class weights, where keys refer to land-use classes, and values to weights. If specified, weighted total supply will be determined. Defaults to None.
            write_non_weighted_result (bool, optional): Indicates if non-weighted total supply be computed. Defaults to True.
        """
        self.printStepInfo('Determining clumped total supply')

        # progress reporting        
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge))                
        current_task = self._get_task("[white]Aggregating clumped supply", total=step_count)

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
        current_task = self._get_task("[white]Determining class diversity", total=step_count)

        with self.progress as p:
            for c in self.cost_thresholds:            
                mtx_diversity_at_cost = self._get_value_matrix()

                for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):                
                    # determine source of list
                    lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"                    
                    mtx_supply = self._get_supply_for_lu_and_cost(lu, lu_type, c)                   
                    mtx_supply[mtx_supply > 0] = 1
                    mtx_diversity_at_cost += mtx_supply
                    p.update(current_task, advance=1)
                
                # export current cost diversity
                self._write_dataset("INDICATORS/diversity_cost_{}.tif".format(c), mtx_diversity_at_cost) 

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
        current_task = self._get_task("[white]Determine class-based flows within cost", step_count)

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
                    p.update(current_task, advance=1)

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

        Args:
            lu_weights (Dict[any,float], optional): Dictionary of land-use class weights, where keys refer to land-use classes, and values to weights. If specified, weighted total supply will be determined. Defaults to None.
            cost_weights (Dict[float, float], optional): Dictionary of cost weights, where keys refer to cost thresholds, and values to weights. If specified, weighted total supply will be determined. Defaults to None.
            write_non_weighted_result (bool, optional): Indicates if non-weighted total supply be computed. Defaults to True.
            write_scaled_result (bool, optional): Indicates if min-max-scaled values should be written as separate outputs. Defaults to True. 
        """
        self.printStepInfo("Averaging supply across costs")
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge))
        current_task = self._get_task("[white]Averaging supply", total=step_count)

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

        Args:
            cost_weights (Dict[float, float], optional): Dictionary of cost weights, where keys refer to cost thresholds, and values to weights. If specified, weighted total supply will be determined. Defaults to None.
            write_non_weighted_result (bool, optional): Indicates if non-weighted total supply be computed. Defaults to True.
            write_scaled_result (bool, optional): Indicates if min-max-scaled values should be written as separate outputs. Defaults to True. 
        """
        self.printStepInfo("Averaging diversity across costs")
        step_count = len(self.cost_thresholds)
        current_task = self._get_task("[white]Averaging diversity", total=step_count)

        with self.progress as p:

            # result raster
            if write_non_weighted_result:
                average_diversity = self._get_value_matrix()
            if cost_weights is not None:
                cost_weighted_average_diversity = self._get_value_matrix()

            # iterate over cost thresholds and aggregate cost-specific diversities into result
            for c in self.cost_thresholds:
                mtx_current_diversity = self._read_band("INDICATORS/diversity_cost_{}.tif".format(c)) 
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

        Args:
            cost_weights (Dict[float, float], optional): Dictionary of cost weights, where keys refer to cost thresholds, and values to weights. If specified, weighted total supply will be determined. Defaults to None.
            write_non_weighted_result (bool, optional): Indicates if non-weighted total supply be computed. Defaults to True.
            write_scaled_result (bool, optional): Indicates if min-max-scaled values should be written as separate outputs. Defaults to True. 
        """
        self.printStepInfo("Averaging beneficiaries across costs")
        step_count = len(self.cost_thresholds)
        current_task = self._get_task("[white]Averaging beneficiaries", total=step_count)

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

        Args:
            cost_weights (Dict[float, float], optional): Dictionary of cost weights, where keys refer to cost thresholds, and values to weights. If specified, weighted total supply will be determined. Defaults to None.
            write_non_weighted_result (bool, optional): Indicates if non-weighted total supply be computed. Defaults to True.
            write_scaled_result (bool, optional): Indicates if min-max-scaled values should be written as separate outputs. Defaults to True. 
        """
        self.printStepInfo("Averaging flow across costs")        
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge)) 
        current_task = self._get_task("[white]Averaging flow across costs", total=step_count)

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
        lu_progress = self._get_task("[white]Per-capita assessment", total=step_count)

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




    def cost_to_closest(self, threshold_masking: bool = True, distance_threshold: float = 25, builtup_masking: bool = False, write_scaled_result: bool = True) -> None:
        """Determines cost to closest entities of each land-use class, and determines averaged cost to closest.

        Args:
            threshold_masking (bool, optional): Should a maximum distance be considered for assessing costs to closest? Defaults to True.
            distance_threshold (float, optional): Threshold for distance-cost-based masking. Defaults to 25.
            builtup_masking (bool, optional): Should result be restricted to built-up land-use pixels?. Defaults to False.
            write_scaled_result (bool, optional): Indicates if min-max-scaled values should be written as separate outputs. Defaults to True. 
        """
        self.printStepInfo("Assessing cost to closest")
        included_lu_classes = self.lu_classes_recreation_patch + self.lu_classes_recreation_edge
        step_count = len(included_lu_classes)         
        current_task = self._get_task("[white]Assessing cost to closest", total=step_count)

        with self.progress as p:
            # get built-up layer 
            if threshold_masking:
                print(Fore.YELLOW + Style.BRIGHT + "APPLYING THRESHOLD MASKING" + Style.RESET_ALL)

            if builtup_masking:
                mtx_builtup = self._read_band('MASKS/built-up.tif')

            # raster for average result
            mtx_average_cost = self._get_value_matrix()
            mtx_lu_cost_count_considered = self._get_value_matrix()

            for lu in included_lu_classes:
                # import pre-computed proximity raster
                mtx_proximity = self._read_band("PROX/dr_{}.tif".format(lu))
                # mask out values that are greater than upper_threshold as they are considered irrelevant, if requested by user
                if threshold_masking:
                    # fill higher values with upper threshold
                    mtx_proximity[mtx_proximity > distance_threshold] = 0 # here, 0 is equal to the lsm nodata value
                
                if builtup_masking:
                    # intersect with built-up to determine closest costs
                    # a simple multiplication should result in                     
                    mtx_proximity = mtx_proximity * mtx_builtup
                               
                # write result to disk
                self._write_dataset('COSTS/minimum_cost_{}.tif'.format(lu), mtx_proximity)
                # add to result
                mtx_average_cost += mtx_proximity

                # now mask values that are not 0 with 1, to determine for each pixel the number of costs considered
                mtx_proximity[mtx_proximity != 0] = 1
                mtx_lu_cost_count_considered += mtx_proximity
                
                p.update(current_task, advance=1)

        # export average cost grid
        # prior, determine actual average. here, consider per each pixel the number of grids added.
        mtx_average_cost = np.divide(mtx_average_cost, mtx_lu_cost_count_considered, where=mtx_lu_cost_count_considered > 0)
        self._write_dataset('INDICATORS/non_weighted_avg_cost.tif', mtx_average_cost)
        if write_scaled_result:
            # apply min-max scaling
            scaler = MinMaxScaler()
            mtx_average_cost = 1-scaler.fit_transform(mtx_average_cost.reshape([-1,1]))
            self._write_dataset('INDICATORS/scaled_non_weighted_avg_cost.tif', mtx_average_cost.reshape(self.lsm_mtx.shape))

        
        # done
        self.taskProgressReportStepCompleted()



    #
    # Helper functions
    #
    #


    def _read_dataset(self, file_name: str, band: int = 1, nodata_values: List[float] = [0], is_scenario_specific:bool = True, nodata_fill_value = None) -> Tuple[rasterio.DatasetReader, np.ndarray, np.ndarray]:
        """Read a dataset and return reference to the dataset, values, and boolean mask of nodata values.

        Args:
            file_name (str): Filename of dataset to be read.
            band (int, optional): Band to be read. Defaults to 1.
            nodata_values (List[float], optional): List of values indicating nodata. Defaults to [0].
            is_scenario_specific (bool, optional): Indicates if the specified datasource located in a scenario-specific subfolder (True) or at the data path root (False). Defaults to True.
            nodata_fill_value (float, optional): If set to a value, nodata values of the raster to be imported will be filled up with the specified value.           

        Returns:
            Tuple[rasterio.DatasetReader, np.ndarray, np.ndarray]: Dataset, data matrix, and mask of nodata values.
        """
        path = "{}/{}".format(self.data_path, file_name) if not is_scenario_specific else "{}/{}/{}".format(self.data_path, self.root_path, file_name)
        if self.verbose_reporting:
            print(Fore.WHITE + Style.DIM + "    READING {}".format(path) + Style.RESET_ALL)
        
        rst_ref = rasterio.open(path)
        band_data = rst_ref.read(band)

        # attempt replacement of nodata with desired fill value
        fill_value = self.nodata_value if nodata_fill_value is None else nodata_fill_value
        for nodata_value in nodata_values:

            # replace only if not the same values!
            if fill_value != nodata_value:
                if self.verbose_reporting:                
                    print(Fore.YELLOW + Style.DIM + "    REPLACING NODATA VALUE={} WITH FILL VALUE={}".format(nodata_value, fill_value) + Style.RESET_ALL) 
                band_data = np.where(band_data==nodata_value, fill_value, band_data)

        # determine nodata mask AFTER potential filling of nodata values        
        nodata_mask = np.isin(band_data, nodata_values, invert=False)                
        return rst_ref, band_data, nodata_mask
        
    def _read_band(self, file_name: str, band: int = 1, is_scenario_specific: bool = True) -> np.ndarray:
        """Read a raster band. 

        Args:
            file_name (str): Filename of dataset to be read.
            band (int, optional): Band to be read. Defaults to 1.
            is_scenario_specific (bool, optional): Indicates if the specified datasource located in a scenario-specific subfolder (True) or at the data path root (False). Defaults to True.

        Returns:
            np.ndarray: _description_
        """
        path = "{}/{}".format(self.data_path, file_name) if not is_scenario_specific else "{}/{}/{}".format(self.data_path, self.root_path, file_name)
        rst_ref, band_data, nodata_mask = self._read_dataset(file_name=file_name, band=band, is_scenario_specific=is_scenario_specific)
        return band_data
    
    def _write_dataset(self, file_name: str, outdata: np.ndarray, mask_nodata: bool = True, is_scenario_specific: bool = True, custom_grid_reference: rasterio.DatasetReader = None, custom_nodata_mask: np.ndarray = None) -> None:        
        """Write a dataset to disk.

        Args:
            file_name (str): Name of file to be written.
            outdata (np.ndarray): Values to be written.
            mask_nodata (bool, optional): Indicates if nodata values should be masked using default or custom nodata mask (True) or not (False). Defaults to True. Uses custom nodata mask if specified.
            is_scenario_specific (bool, optional): Indicates whether file should be written in a scenario-specific subfolder (True) or in the data path root (False). Defaults to True.
            custom_grid_reference (rasterio.DatasetReader, optional): Custom grid reference (i.e., crs, transform) to be used. If not specified, uses default land-use grid crs and transform.
            custom_nodata_mask (np.ndarray, optional): Custom nodata mask to apply if mask_nodata is set to True.
        """

        custom_grid_reference = custom_grid_reference if custom_grid_reference is not None else self.lsm_rst

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
            crs=custom_grid_reference.crs,
            transform=custom_grid_reference.transform
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

    def _get_value_matrix(self, fill_value: float = 0, shape: Tuple[int, int] = None) -> np.ndarray:
        """Return array with specified fill value. 

        Args:
            fill_value (float, optional): Fill value. Defaults to 0.
            shape (Tuple[int, int], optional): Shape of the matrix to be returned.

        Returns:
            np.ndarray: Filled array.
        """
        
        # determine parameters based on specified method arguments
        rst_dtype = self.lsm_mtx.dtype if self.dtype is None else self.dtype
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
    
    def _moving_window_generic(self, data_mtx: np.ndarray, kernel_func: Callable[[np.ndarray], float], kernel_size: int, kernel_shape: str = 'circular') -> np.ndarray:
        """Conduct a moving window operation with specified kernel shape and kernel size on an array.

        Args:
            data_mtx (np.ndarray): Input array
            kernel_func (Callable[[np.ndarray], float]): Callable for aggregation/Summarization of values in kernel window.
            kernel_size (int): Size of kernel (total with for squared kernel window, kernel diameter for circular kernel window).
            kernel_shape (str, optional): Kernel shape: Circular kernel (circular) or squared/rectangular kernel (rect). Defaults to 'circular'.

        Returns:
            np.ndarray: Output array
        """
        # define properties of result matrix
        # for the moment, use the dtype set by user
        target_dtype = self.lsm_mtx.dtype if self.dtype is None else self.dtype

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
    
    

    


    def taskProgressReportStepCompleted(self):
        if self.task_assess_map_units is not None:
            self.progress.update(self.task_assess_map_units, advance=1)
        else:
            self.printStepCompleteInfo()

    def _runsAsStandalone(self):
        return True if self.task_assess_map_units is None else False

    def clean_temporary_files(self):
        """Clean temporary files from the TMP folder.
        """
        tmp_path = os.path.join(self.data_path, self.root_path, 'TMP')
        tmp_files = [os.path.join(self.data_path, self.root_path, 'TMP', f) for f in listdir(tmp_path) if isfile(join(tmp_path, f))]
        
        step_count = len(tmp_files)
        current_task = self._get_task('[red]Removing temporary files', total=step_count)
        
        for f in tmp_files:
            os.remove(f)
            self.progress.update(current_task, advance=1)
        
        self.printStepCompleteInfo(msg = "TEMPORARY FILES CLEANED")
        
        
    
    
    

    
    







    

    


