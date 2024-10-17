#####################
# Author: Sebastian Scheuer (sebastian.scheuer@geo.hu-berlin.de, seb.scheuer@outlook.de)
# Humboldt-Universität zu Berlin
#
#
# The indicators determined based on this script is really a simplification of a more-detailed approach
# adapted to the LSM and WP7 requirements to obtain indicators in a scaled manner.
#
# 
#
#####################


import os
import uuid

from colorama import init as colorama_init
from colorama import Fore, Back, Style
from rich.progress import Progress, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn, BarColumn
from contextlib import nullcontext

import rasterio
from scipy import ndimage
import numpy as np
from skimage.draw import disk
from sklearn.preprocessing import MinMaxScaler
import distancerasters as dr

from typing import Tuple, List, Callable, Dict

colorama_init()

class LsmGridRecreationModeller:

    # some status variables
    verbose_reporting = False
    scenario_intialized = False

    # working directory
    data_path = None

    # scenario name and corresponding population and lsm filenames
    scenario_name = "current"
    lsm_fileName = None

    # this stores the lsm map as reference map
    lsm_rst = None
    lsm_mtx = None
    lsm_nodataMask = None

    # define relevant recreation patch and edge classes, cost thresholds, etc.
    lu_classes_recreation_edge = []
    lu_classes_recreation_patch = []
    lu_classes_builtup = []
    cost_thresholds = []
    
    # the following items are computed 
    clump_slices = []

    # progress reporting
    progress = None     
    task_assess_map_units = None

    def __init__(self, data_path: str):
        os.system('cls' if os.name == 'nt' else 'clear')
        self.data_path = data_path                
                    
    def make_environment(self) -> None:
        """Create required subfolders for raster files in the current scenario folder.
        """
        # create directories, if needed
        dirs_required = ['DEMAND', 'MASKS', 'SUPPLY', 'INDICATORS', 'TMP', 'FLOWS', 'CLUMPS_LU', 'PROX', 'COSTS']
        for d in dirs_required:
            cpath = "{}/{}/{}".format(self.data_path, self.scenario_name, d)
            if not os.path.exists(cpath):
                os.makedirs(cpath)

    def printStepInfo(self, msg):
        print(Fore.CYAN + Style.DIM + msg.upper() + Style.RESET_ALL)
    def printStepCompleteInfo(self):
        print(Fore.WHITE + Back.GREEN + "COMPLETED" + Style.RESET_ALL)

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

    def set_params(self, paramType, paramValue):
        if paramType == 'classes.edge':
            self.lu_classes_recreation_edge = paramValue
        elif paramType == 'classes.patch':
            self.lu_classes_recreation_patch = paramValue
        elif paramType == 'classes.builtup':
            self.lu_classes_builtup = paramValue
        elif paramType == 'costs':
            self.cost_thresholds = paramValue
      
    def set_land_use_map(self, root_path: str, land_use_file: str) -> None:
        """Specify data sources for a given scenario, and import land-use raster file. Note that projection and resolution of raster files must match. 

        Args:
            scenario_name (str): Name of a scenario, i.e., subfolder within root of data path.
            lu_file (str): Name of the land-use raster file for the given scenario.           
        """
        self.scenario_name = root_path
        self.lsm_fileName = land_use_file
        
        # check if folders are properly created in current scenario workspace
        self.make_environment()         
        
        # import lsm
        self.lsm_rst, self.lsm_mtx, self.lsm_nodataMask = self._read_dataset(self.lsm_fileName)

    def assess_map_units(self, compute_proximities: bool = False) -> None:
        """Determine basic data and conduct basic operations on land-use classes, including occurence masking, edge detection, aggregation of built-up classes and population disaggregation, determination of beneficiaries within given cost thresholds, computation of proximities to land-uses (if requested), and land-use class-specific total supply.   

        Args:
            compute_proximities (bool, optional): Compute proximities to land-uses. This is very computationally heavy. Defaults to False.
        """
        self.progress = self.get_progress_bar()
        step_count = 7 if compute_proximities else 6
        self.task_assess_map_units = self.progress.add_task('[red]Assessing recreational potential', total=step_count)
        with self.progress as p:
            
            # detecting clumps
            self.detect_clumps()
                        
            # mask land-uses and detect edges on relevant classes
            self.mask_landuses()    
            self.detect_edges()

            # work on population disaggregation
            self.disaggregate_population()            
            self.beneficiaries_within_cost()

            # determine raster-based proximities as cost-to-closest
            if compute_proximities:
                self.compute_distance_rasters()        
            
            # determine supply per class
            self.class_total_supply()

        self.printStepCompleteInfo()
        self.task_assess_map_units = None


    def taskProgressReportStepCompleted(self):
        if self.task_assess_map_units is not None:
            self.progress.update(self.task_assess_map_units, advance=1)
        else:
            self.printStepCompleteInfo()

    def _runsAsStandalone(self):
        return True if self.task_assess_map_units is None else False




    #
    # The following classes will be called from asses_map_units. 
    # They will disaggregate population and determine clumped land-use class supplies.
    # Layers written will be specific to given costs.
    #
        
    def detect_clumps(self):

        self.printStepInfo("Detecting clumps")
        clump_connectivity = np.full((3,3), 1)
        rst_clumps = self._get_value_matrix()
        nr_clumps = ndimage.label(self.lsm_mtx, structure=clump_connectivity, output=rst_clumps)
        print(Fore.YELLOW + Style.BRIGHT + "{} CLUMPS FOUND".format(nr_clumps) + Style.RESET_ALL)
        self._write_dataset("MASKS/clumps.tif", rst_clumps)        
        # make slices to speed-up window operations
        self.clump_slices = ndimage.find_objects(rst_clumps.astype(np.int64))        
        
        # done
        self.taskProgressReportStepCompleted()

    def mask_landuses(self):
        # mask classes of interest into a binary raster to indicate presence/absence of recreational potential
        # we require this for all classes relevant to processing: patch and edge recreational classes, built-up classes
        self.printStepInfo("CREATING LAND-USE MASKS")
        classes_for_masking = self.lu_classes_recreation_edge + self.lu_classes_recreation_patch + self.lu_classes_builtup 
        current_task = self._get_task('[white]Masking land-uses', total=len(classes_for_masking))
        
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
    
    def detect_edges(self):
        # determine edge pixels of edge-only classes such as water opportunities
        if(len(self.lu_classes_recreation_edge) > 0):
            
            self.printStepInfo("Detecting edges")
            current_task = self._get_task("[white]Detecting edges", total=len(self.lu_classes_recreation_edge))
            
            for lu in self.lu_classes_recreation_edge:            
                inputMaskFileName = "MASKS/mask_{}.tif".format(lu)    
                mtx_mask = self._read_band(inputMaskFileName)            
                # apply a 3x3 rectangular sliding window to determine pixel value diversity in window
                rst_edgePixelDiversity = self._moving_window(mtx_mask, self._kernel_diversity, 3, 'rect') 
                rst_edgePixelDiversity = rst_edgePixelDiversity - 1
                mtx_mask = mtx_mask * rst_edgePixelDiversity
                self._write_dataset("MASKS/edges_{}.tif".format(lu), mtx_mask)
                del rst_edgePixelDiversity
                self.progress.update(current_task, advance=1)
        
            # done
            self.taskProgressReportStepCompleted()

    def compute_distance_rasters(self, lu_classes: List[int] = None, assess_builtup: bool = False) -> None:
        """Generate proximity rasters to land-use classes.

        Args:
            lu_classes (List[int], optional): List of integers, i.e., land-use classes to assess. Defaults to None.
            assess_builtup (bool, optional): Assesses proximities to built-up, if true. Defaults to False.
        """
        self.printStepInfo("Computing distance rasters")
        # determine proximity outward from relevant lu classes, including built-up
        classes_for_proximity_calculation = (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge) if lu_classes is None else lu_classes

        # import the clumps raster
        rst_clumps = self._read_band("MASKS/clumps.tif")
        # detect slices here, to avoid having to recall detect_clumps each time we want to do proximity computations
        # note: todo: this should also be changed elsewhere where we rely on clumps in later evaluation steps.
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

                        # now all pixels outside of clump should be zeroed, and we can determine proximity on the subset of the full raster
                        sliced_dr = dr.DistanceRaster(sliced_lu_mtx, progress_bar=False)
                        sliced_dr = sliced_dr.dist_array
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

    def disaggregate_population(self, population_grid: str, write_scaled_result: bool = True) -> None:
        """Aggregates built-up land-use classes into a single raster of built-up areas, and intersects built-up with the scenario-specific population grid to provide disaggregated population.

        Args:
            population_grid (str): Name of the population raster file to be used for disaggregation.
            write_scaled_result (bool, optional): Export min-max scaled result, if True. Defaults to True.
        """
        self.printStepInfo("Disaggregating population to built-up")

        current_task = self._get_task("[white]Population disaggregation", total=len(self.lu_classes_builtup))

        with self.progress if self._runsAsStandalone() else nullcontext() as bar:

            mtx_builtup = self._get_value_matrix()            
            mtx_pop = self._read_band(population_grid)
            
            for lu in self.lu_classes_builtup:
                rst_mtx = self._read_band('MASKS/mask_{}.tif'.format(lu))
                mtx_builtup += rst_mtx                   
                self.progress.update(current_task, advance=1)                 
                
            # write built-up raster to disk
            self._write_dataset("MASKS/built-up.tif", mtx_builtup)

            # now disaggregate poopulation by intersect                
            # multiply residential built-up pixels with pop raster        
            mtx_builtup = mtx_builtup * mtx_pop
            # write pop raster to disk
            self._write_dataset("DEMAND/disaggregated_population.tif", mtx_builtup)
            if write_scaled_result:
                scaler = MinMaxScaler()
                mtx_builtup = scaler.fit_transform(mtx_builtup.reshape([-1,1]))
                self._write_dataset("DEMAND/scaled_disaggregated_population.tif", mtx_builtup.reshape(self.lsm_mtx.shape))
        
        # done   
        self.taskProgressReportStepCompleted()

        
    def beneficiaries_within_cost(self):        
        self.printStepInfo("Determining beneficiaries within costs")
        
        step_count = len(self.cost_thresholds) * len(self.clump_slices)
        current_task = self._get_task("[white]Determining beneficiaries", total=step_count)

        # get relevant input data
        mtx_disaggregated_population = self._read_band("DEMAND/disaggregated_population.tif")        
        # also beneficiaries need to be clumped
        rst_clumps = self._read_band("MASKS/clumps.tif")
        
        for c in self.cost_thresholds:

            mtx_pop_within_cost = self._get_value_matrix()

            # now operate over clumps, in order to safe some computational time
            for patch_idx in range(len(self.clump_slices)):
                obj_slice = self.clump_slices[patch_idx]
                obj_label = patch_idx + 1

                # get slice from land-use mask
                sliced_pop_mtx = mtx_disaggregated_population[obj_slice].copy() 
                sliced_clump_mtx = rst_clumps[obj_slice]

                # properly mask out current object
                obj_mask = np.isin(sliced_clump_mtx, [obj_label], invert=False)
                sliced_pop_mtx[~obj_mask] = 0

                # now all pixels outside of clump should be zeroed, and we can determine total supply within sliding window
                sliding_pop = self._moving_window(sliced_pop_mtx, self._kernel_sum, c)
                sliding_pop[~obj_mask] = 0
                mtx_pop_within_cost[obj_slice] += sliding_pop
                
                del sliding_pop
                del sliced_pop_mtx
                # progress reporting
                self.progress.update(current_task, advance=1)

            self._write_dataset("DEMAND/beneficiaries_within_cost_{}.tif".format(c), mtx_pop_within_cost)

        # done
        self.taskProgressReportStepCompleted()

    def class_total_supply(self):
        
        # for each recreation patch class and edge class, determine total supply within cost windows
        # do this for each clump, i.e., operate only on parts of masks corresponding to clumps, ignore patches/edges external to each clump
        self.printStepInfo("Determining clumped supply per class")
        # clumps are required to properly mask islands
        rst_clumps = self._read_band("MASKS/clumps.tif")

        step_count = len(self.clump_slices) * (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch)) * len(self.cost_thresholds)
        current_task = self._get_task("[white]Determining clumped supply", total=step_count)

        for c in self.cost_thresholds:            
                        
            for lu in self.lu_classes_recreation_patch:
                # process supply of current class 
                lu_supply_mtx = self._class_total_supply_for_lu_and_cost("MASKS/mask_{}.tif".format(lu), rst_clumps, c, current_task)                            
                # export current cost
                self._write_dataset("SUPPLY/totalsupply_class_{}_cost_{}_clumped.tif".format(lu, c), lu_supply_mtx)                
                
            for lu in self.lu_classes_recreation_edge:
                # process supply of current class 
                lu_supply_mtx = self._class_total_supply_for_lu_and_cost("MASKS/edges_{}.tif".format(lu), rst_clumps, c, current_task)                            
                # export current cost
                self._write_dataset("SUPPLY/totalsupply_edge_class_{}_cost_{}_clumped.tif".format(lu, c), lu_supply_mtx)                

        # done
        self.taskProgressReportStepCompleted()




    def _class_total_supply_for_lu_and_cost(self, mask_path, rst_clumps, cost, progress_task = None):
        
        # grid to store lu supply 
        lu_supply_mtx = self._get_value_matrix()
        # get land-use current mask
        full_lu_mtx = self._read_band(mask_path)

        # now operate over clumps, in order to safe some computational time
        for patch_idx in range(len(self.clump_slices)):
            obj_slice = self.clump_slices[patch_idx]
            obj_label = patch_idx + 1

            # get slice from land-use mask
            sliced_lu_mtx = full_lu_mtx[obj_slice].copy() 
            sliced_clump_mtx = rst_clumps[obj_slice]

            # properly mask out current object
            obj_mask = np.isin(sliced_clump_mtx, [obj_label], invert=False)
            sliced_lu_mtx[~obj_mask] = 0

            # now all pixels outside of clump should be zeroed, and we can determine total supply within sliding window
            sliding_supply = self._moving_window(sliced_lu_mtx, self._kernel_sum, cost)
            sliding_supply[~obj_mask] = 0
            lu_supply_mtx[obj_slice] += sliding_supply
            
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
    
    def aggregate_class_total_supply(self, lu_weights: Dict[int,float] = None, write_non_weighted_result: bool = True) -> None:
        """Aggregate total supply of land-use classes within each specified cost threshold. A weighting schema may be supplied, in which case a weighted average is determined as the sum of weighted class supply divided by the sum of all weights.

        Args:
            lu_weights (Dict[int,float], optional): Dictionary of land-use class weights, where keys refer to land-use classes, and values to weights. If specified, weighted total supply will be determined. Defaults to None.
            write_non_weighted_result (bool, optional): Indicates if non-weighted total supply be computed. Defaults to True.
        """
        self.printStepInfo('Determining clumped total supply')

        # progress reporting        
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge))                
        current_task = self._get_task("[white]Aggregating clumped supply", total=step_count)

        with self.progress as p:

            for c in self.cost_thresholds:
                # get aggregation for current cost threshold
                current_total_supply_at_cost, current_weighted_total_supply_at_cost = self._get_aggregate_class_total_supply_for_cost(c, lu_weights, write_non_weighted_result, current_task)                                           
                
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

    def average_total_supply_across_cost(self, lu_weights: Dict[int, float] = None, cost_weights: Dict[float, float] = None, write_non_weighted_result: bool = True, write_scaled_result: bool = True) -> None:
        """Determine the total (recreational) land-use supply averaged across cost thresholds. Weighting of importance of land-uses and weighting of cost may be applied. 
           If either weighting schema (land-use classes or costs) is supplied, the total supply is determined as weighted average, i.e., the weighted sum of land-use class-specific supply, divided by the sum of weights.
           Potential combinations, i.e., land-use and subsequently cost-based weighting, are considered if both weighting schemas are supplied.

        Args:
            lu_weights (Dict[int,float], optional): Dictionary of land-use class weights, where keys refer to land-use classes, and values to weights. If specified, weighted total supply will be determined. Defaults to None.
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

            # todo: check clump
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


    def _read_dataset(self, file_name: str, band: int = 1, nodata_values: List[float] = [0], is_scenario_specific:bool = True) -> Tuple[rasterio.DatasetReader, np.ndarray, np.ndarray]:
        """Read a dataset and return reference to the dataset, values, and boolean mask of nodata values.

        Args:
            file_name (str): Filename of dataset to be read.
            band (int, optional): Band to be read. Defaults to 1.
            nodata_values (List[float], optional): List of values indicating nodata. Defaults to [0].
            is_scenario_specific (bool, optional): Indicates if the specified datasource located in a scenario-specific subfolder (True) or at the data path root (False). Defaults to True.

        Returns:
            Tuple[rasterio.DatasetReader, np.ndarray, np.ndarray]: Dataset, data matrix, and mask of nodata values.
        """
        path = "{}/{}".format(self.data_path, file_name) if not is_scenario_specific else "{}/{}/{}".format(self.data_path, self.scenario_name, file_name)
        if self.verbose_reporting:
            print(Fore.WHITE + Style.DIM + "READING {}".format(path) + Style.RESET_ALL)
        rst_ref = rasterio.open(path)
        band_data = rst_ref.read(band)
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
        path = "{}/{}".format(self.data_path, file_name) if not is_scenario_specific else "{}/{}/{}".format(self.data_path, self.scenario_name, file_name)
        if self.verbose_reporting:
            print(Fore.WHITE + Style.DIM + "READING {}".format(path) + Style.RESET_ALL)
        rst_ref = rasterio.open(path)
        band_data = rst_ref.read(band)
        return band_data
    
    def _write_dataset(self, file_name: str, outdata: np.ndarray, mask_nodata: bool = True, is_scenario_specific: bool = True) -> None:        
        """Write a dataset to disk.

        Args:
            file_name (str): Name of file to be written.
            outdata (np.ndarray): Values to be written.
            mask_nodata (bool, optional): Indicates if nodata values should be masked using default nodata mask (True) or not (False). Defaults to True.
            is_scenario_specific (bool, optional): Indicates whether file should be written in a scenario-specific subfolder (True) or in the data path root (False). Defaults to True.
        """
        path = "{}/{}".format(self.data_path, file_name) if not is_scenario_specific else "{}/{}/{}".format(self.data_path, self.scenario_name, file_name)
        if self.verbose_reporting:
            print(Fore.YELLOW + Style.DIM + "WRITING {}".format(path) + Style.RESET_ALL)

        if mask_nodata is True:
            outdata[self.lsm_nodataMask] = 0    

        with rasterio.open(
            path,
            mode="w",
            driver="GTiff",
            height=self.lsm_mtx.shape[0],
            width=self.lsm_mtx.shape[1],
            count=1,
            dtype=self.lsm_mtx.dtype,
            crs=self.lsm_rst.crs,
            transform=self.lsm_rst.transform
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

    def _get_value_matrix(self, fill_value: float = 0) -> np.ndarray:
        """Return array with specified fill value. 

        Args:
            fill_value (float, optional): Fill value. Defaults to 0.

        Returns:
            np.ndarray: Filled array.
        """
        rst_new = np.full(shape=self.lsm_mtx.shape, fill_value=fill_value, dtype=self.lsm_mtx.dtype)
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
    
    def _kernel_diversity(self, subarr: np.ndarray) -> float:
        """Determine the number of unique elements in a kernel window.

        Args:
            subarr (np.ndarray): Kernel.

        Returns:
            int: Number of unique elements in kernel.
        """
        return len(set(subarr))

    def _moving_window(self, data_mtx: np.ndarray, kernel_func: Callable[[np.ndarray], float], kernel_size: int, kernel_shape: str = 'circular') -> np.ndarray:
        """Conduct a moving window operation with specified kernel shape and kernel size on an array.

        Args:
            data_mtx (np.ndarray): Input array
            kernel_func (Callable[[np.ndarray], float]): Callable for aggregation/Summarization of values in kernel window.
            kernel_size (int): Size of kernel (total with for squared kernel window, kernel diameter for circular kernel window).
            kernel_shape (str, optional): Kernel shape: Circular kernel (circular) or squared/rectangular kernel (rect). Defaults to 'circular'.

        Returns:
            np.ndarray: Output array
        """
        # make kernel
        kernel = self._get_circular_kernel(kernel_size) if kernel_shape == 'circular' else np.full((kernel_size, kernel_size), 1)
        # create result mtx as memmap
        mtx_res = np.memmap("{}/{}/TMP/{}".format(self.data_path, self.scenario_name, uuid.uuid1()), dtype=data_mtx.dtype, mode='w+', shape=data_mtx.shape) 
        # apply moving window over input mtx
        ndimage.generic_filter(data_mtx, kernel_func, footprint=kernel, output=mtx_res, mode='constant', cval=0)
        mtx_res.flush()
        return mtx_res
    
    def _get_aggregate_class_total_supply_for_cost(self, cost, lu_weights = None, write_non_weighted_result = True, write_scaled_result = True, task_progress = None):                        
        
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
    
    

    
        
    
    
    

    
    







    

    


