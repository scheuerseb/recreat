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


import numpy as np
import rasterio
from scipy import ndimage
from skimage.draw import disk
from alive_progress import alive_bar
import os
import uuid
from colorama import init as colorama_init
from colorama import Fore, Back, Style
from rich.progress import Progress, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn, BarColumn
import distancerasters as dr
from contextlib import nullcontext

colorama_init()

class LsmGridRecreationModeller:

    # some status variables
    verbose_reporting = False
    scenario_intialized = False

    # working directory
    dataPath = None

    # scenario name and corresponding population and lsm filenames
    scenario_name = "current"
    pop_fileName = None
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

    def __init__(self, dataPath):
        os.system('cls' if os.name == 'nt' else 'clear')
        self.dataPath = dataPath                
                    
    def make_environment(self):
        # create directories, if needed
        dirs_required = ['DEMAND', 'MASKS', 'SUPPLY', 'INDICATORS', 'TMP', 'FLOWS', 'CLUMPS_LU', 'PROX', 'COSTS']
        for d in dirs_required:
            cpath = "{}/{}/{}".format(self.dataPath, self.scenario_name, d)
            if not os.path.exists(cpath):
                os.makedirs(cpath)

    def printStepInfo(self, msg):
        print(Fore.CYAN + Style.DIM + msg.upper() + Style.RESET_ALL)
    def printStepCompleteInfo(self):
        print(Fore.WHITE + Back.GREEN + "COMPLETED" + Style.RESET_ALL)

    def new_progress(self, task_description, step_count):
        self.progress = self.get_progress_bar()
        task_new = self.progress.add_task(task_description, total=step_count)
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
      
    def set_scenario(self, scenario_name, lsm_filename, population_filename):
        self.scenario_name = scenario_name
        self.lsm_fileName = lsm_filename
        self.pop_fileName = population_filename   
        
        # check if folders are properly created in current scenario workspace
        self.make_environment()         
        
        # import lsm
        self.lsm_rst, self.lsm_mtx, self.lsm_nodataMask = self.read_dataset(self.lsm_fileName)

    def assess_map_units(self, compute_proximities = False):       
        
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
                self.compute_distance_rasters(standalone = False)        
            
            # determine supply per class
            self.class_total_supply()

        self.printStepCompleteInfo()


    def advanceStepTotal(self):
        self.progress.update(self.task_assess_map_units, advance=1)




    #
    # The following classes will be called from asses_map_units. 
    # They will disaggregate population and determine clumped land-use class supplies.
    # Layers written will be specific to given costs.
    #
        
    def detect_clumps(self):
        self.printStepInfo("Detecting clumps")
        clump_connectivity = np.full((3,3), 1)
        rst_clumps = self.get_value_matrix()
        nr_clumps = ndimage.label(self.lsm_mtx, structure=clump_connectivity, output=rst_clumps)
        print(Fore.YELLOW + Style.BRIGHT + "{} CLUMPS FOUND".format(nr_clumps) + Style.RESET_ALL)
        self.write_dataset("MASKS/clumps.tif", rst_clumps)        
        # make slices to speed-up window operations
        self.clump_slices = ndimage.find_objects(rst_clumps.astype(np.int64))        
        # done
        self.advanceStepTotal()

    def mask_landuses(self):
        # mask classes of interest into a binary raster to indicate presence/absence of recreational potential
        # we require this for all classes relevant to processing: patch and edge recreational classes, built-up classes
        self.printStepInfo("CREATING LAND-USE MASKS")
        classes_for_masking = self.lu_classes_recreation_edge + self.lu_classes_recreation_patch + self.lu_classes_builtup 
        # step progress
        task_masking = self.progress.add_task('[white]Masking land-uses', total=len(classes_for_masking))
        for lu in classes_for_masking:
            current_lu_mask = self.lsm_mtx.copy()
            # make mask for relevant pixels
            mask = np.isin(current_lu_mask, [lu], invert=False)
            # mask with binary values 
            current_lu_mask[mask] = 1
            current_lu_mask[~mask] = 0
            self.write_dataset("MASKS/mask_{}.tif".format(lu), current_lu_mask)
            self.progress.update(task_masking, advance=1)

        # done    
        self.advanceStepTotal()
    
    def detect_edges(self):
        # determine edge pixels of edge-only classes such as water opportunities
        if(len(self.lu_classes_recreation_edge) > 0):
            self.printStepInfo("Detecting edges")
            task_edges = self.progress.add_task("[white]Detecting edges", total=len(self.lu_classes_recreation_edge))
            for lu in self.lu_classes_recreation_edge:            
                inputMaskFileName = "MASKS/mask_{}.tif".format(lu)    
                mtx_mask = self.read_band(inputMaskFileName)            
                # apply a 3x3 rectangular sliding window to determine pixel value diversity in window
                rst_edgePixelDiversity = self.moving_window(mtx_mask, self.kernel_diversity, 3, 'rect') 
                rst_edgePixelDiversity = rst_edgePixelDiversity - 1
                mtx_mask = mtx_mask * rst_edgePixelDiversity
                self.write_dataset("MASKS/edges_{}.tif".format(lu), mtx_mask)
                del rst_edgePixelDiversity
                self.progress.update(task_edges, advance=1)
        
        # done
        self.advanceStepTotal()

    def compute_distance_rasters(self, standalone = True, lu_classes = None, assess_builtup = False):
        self.printStepInfo("Computing distance rasters")
        # determine proximity outward from relevant lu classes, including built-up
        classes_for_proximity_calculation = (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge) if lu_classes is None else lu_classes
        
        step_count = len(classes_for_proximity_calculation)
        if assess_builtup:
            step_count += 1

        # if standalone, create new progress bar, otherwise use existing bar, and create task
        if standalone:
            current_task = self.new_progress("[white]Computing distance rasters", total=step_count)
        else:
            current_task = self.progress.add_task("[white]Computing distance rasters", total=step_count)

        # iterate over relevant classes
        with self.progress if standalone else nullcontext() as bar:
            for lu in classes_for_proximity_calculation:
                lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
                src_mtx = self.read_band('MASKS/mask_{}.tif'.format(lu) if lu_type == "patch" else 'MASKS/edges_{}.tif'.format(lu))
                my_dr = dr.DistanceRaster(src_mtx, progress_bar=False)
                self.write_dataset("PROX/dr_{}.tif".format(lu), my_dr.dist_array)

                if standalone:
                    bar.update(current_task, advance=1)
                else:
                    self.progress.update(current_task, advance=1)

            if assess_builtup:
                src_mtx = self.read_band('MASKS/built-up.tif') 
                my_dr = dr.DistanceRaster(src_mtx, progress_bar=False)
                self.write_dataset("PROX/dr_built-up.tif".format(lu), my_dr.dist_array)

                if standalone:
                    bar.update(current_task, advance=1)
                else:
                    self.progress.update(current_task, advance=1)
 
        
        # done
        if not standalone:
            self.advanceStepTotal()
        else:
            self.printStepCompleteInfo()

    def disaggregate_population(self):
        self.printStepInfo("Disaggregating population to built-up")
        task_pop = self.progress.add_task("[white]Population disaggregation", total=len(self.lu_classes_builtup)+2)
        mtx_builtup = self.get_value_matrix()
        mtx_pop = self.read_band(self.pop_fileName)
        for lu in self.lu_classes_builtup:
            rst_mtx = self.read_band('MASKS/mask_{}.tif'.format(lu))
            mtx_builtup += rst_mtx   
            self.progress.update(task_pop, advance=1)                 
        # write built-up raster to disk
        self.write_dataset("MASKS/built-up.tif", mtx_builtup)
        self.progress.update(task_pop, advance=1)                 
        # multiply residential built-up pixels with pop raster        
        mtx_builtup = mtx_builtup * mtx_pop
        # write pop raster to disk
        self.write_dataset("DEMAND/disaggregated_population.tif", mtx_builtup)
        self.progress.update(task_pop, advance=1) 
        
        # done   
        self.advanceStepTotal()             

    def beneficiaries_within_cost(self):        
        self.printStepInfo("Determining beneficiaries within costs")
        
        step_count = len(self.cost_thresholds) * len(self.clump_slices)
        task_popcost = self.progress.add_task("[white]Determining beneficiaries", total=step_count)

        mtx_disaggregated_population = self.read_band("DEMAND/disaggregated_population.tif")        
        
        # also beneficiaries need to be clumped
        rst_clumps = self.read_band("MASKS/clumps.tif")
        
        for c in self.cost_thresholds:

            mtx_pop_within_cost = self.get_value_matrix()

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
                sliding_pop = self.moving_window(sliced_pop_mtx, self.kernel_sum, c)
                sliding_pop[~obj_mask] = 0
                mtx_pop_within_cost[obj_slice] += sliding_pop
                
                del sliding_pop
                del sliced_pop_mtx
                # progress reporting
                self.progress.update(task_popcost, advance=1)

            self.write_dataset("DEMAND/beneficiaries_within_cost_{}.tif".format(c), mtx_pop_within_cost)

        # done
        self.advanceStepTotal()

    def class_total_supply(self):
        
        # for each recreation patch class and edge class, determine total supply within cost windows
        # do this for each clump, i.e., operate only on parts of masks corresponding to clumps, ignore patches/edges external to each clump
        self.printStepInfo("Determining clumped supply per class")
        # clumps are required to properly mask islands
        rst_clumps = self.read_band("MASKS/clumps.tif")

        step_count = len(self.clump_slices) * (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch)) * len(self.cost_thresholds)
        task_supply = self.progress.add_task("[white]Determining clumped supply", total=step_count)

        for c in self.cost_thresholds:            
                        
            for lu in self.lu_classes_recreation_patch:
                # process supply of current class 
                lu_supply_mtx = self.class_total_supply_for_lu_and_cost("MASKS/mask_{}.tif".format(lu), rst_clumps, c, task_supply)                            
                # export current cost
                self.write_dataset("SUPPLY/totalsupply_class_{}_cost_{}_clumped.tif".format(lu, c), lu_supply_mtx)                
                
            for lu in self.lu_classes_recreation_edge:
                # process supply of current class 
                lu_supply_mtx = self.class_total_supply_for_lu_and_cost("MASKS/edges_{}.tif".format(lu), rst_clumps, c, task_supply)                            
                # export current cost
                self.write_dataset("SUPPLY/totalsupply_edge_class_{}_cost_{}_clumped.tif".format(lu, c), lu_supply_mtx)                

        # done
        self.advanceStepTotal()

    def class_total_supply_for_lu_and_cost(self, mask_path, rst_clumps, cost, progress_task = None):
        
        # grid to store lu supply 
        lu_supply_mtx = self.get_value_matrix()
        # get land-use current mask
        full_lu_mtx = self.read_band(mask_path)

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
            sliding_supply = self.moving_window(sliced_lu_mtx, self.kernel_sum, cost)
            sliding_supply[~obj_mask] = 0
            lu_supply_mtx[obj_slice] += sliding_supply
            
            del sliding_supply
            del sliced_lu_mtx

            if progress_task is not None:
                self.progress.update(progress_task, advance=1)
        
        # done with current iterations. return result
        del full_lu_mtx
        return lu_supply_mtx
    


    #
    # Aggregate lu-class-specific supply within a given cost to total supply within cost.
    # It can be called following assessment of map units.
    # A weighting schema for lu classes can be supplied.
    #
    
    def aggregate_class_total_supply(self, lu_weights = None, write_non_weighted_result = True):

        self.printStepInfo('Determining clumped total supply')

        # progress reporting        
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge))        
        current_task = self.new_progress("[white]Aggregating clumped supply", step_count)

        with self.progress as p:

            for c in self.cost_thresholds:
                # get aggregation for current cost threshold
                current_total_supply_at_cost, current_weighted_total_supply_at_cost = self.get_aggregate_class_total_supply_for_cost(c, lu_weights, write_non_weighted_result, current_task)                                           
                
                # export total for costs, if requested
                if write_non_weighted_result:                
                    self.write_dataset("INDICATORS/totalsupply_cost_{}.tif".format(c), current_total_supply_at_cost)                
                # export weighted total, if applicable
                if lu_weights is not None:                    
                    self.write_dataset("INDICATORS/weighted_totalsupply_cost_{}.tif".format(c), current_weighted_total_supply_at_cost)
                    
        # done
        self.printStepCompleteInfo()

    def get_aggregate_class_total_supply_for_cost(self, cost, lu_weights = None, write_non_weighted_result = True, task_progress = None):                        
        
        current_total_supply_at_cost = None
        current_weighted_total_supply_at_cost = None

        # make grids for the results: zero-valued grids with full lsm extent
        if write_non_weighted_result:
            current_total_supply_at_cost = self.get_value_matrix() 
        if lu_weights is not None:
            current_weighted_total_supply_at_cost = self.get_value_matrix()
        
        for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):                
            # determine source of list
            lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"                    
            lu_supply_mtx = self.get_supply_for_lu_and_cost(lu, lu_type, cost)

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

    def get_supply_for_lu_and_cost(self, lu, lu_type, cost):        
        # make filename
        filename = "SUPPLY/totalsupply_class_{}_cost_{}_clumped.tif".format(lu, cost) if lu_type == 'patch' else "SUPPLY/totalsupply_edge_class_{}_cost_{}_clumped.tif".format(lu, cost)
        # get supply of current class 
        lu_supply_mtx = self.read_band(filename) 
        # return supply
        return lu_supply_mtx
    
    def get_mask_for_lu(self, lu, lu_type):        
        # make filename
        filename = "MASKS/mask_{}.tif".format(lu) if lu_type == 'patch' else "MASKS/edges_{}.tif".format(lu)
        # get mask of current class 
        lu_mask = self.read_band(filename) 
        # return mask
        return lu_mask
    
        
    #
    # Determine diversity of recreational opportunities within cost based on class-specific supply.
    # 

    def class_diversity(self):

        self.printStepInfo("Determining class diversity within costs")        
        
        step_count = (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch)) * len(self.cost_thresholds)        
        current_task = self.new_progress("[white]Determining class diversity", step_count)

        with self.progress as p:

            for c in self.cost_thresholds:            
                mtx_diversity_at_cost = self.get_value_matrix()

                for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):                
                    # determine source of list
                    lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"                    
                    mtx_supply = self.get_supply_for_lu_and_cost(lu, lu_type, c)
                    mtx_supply[mtx_supply > 0] = 1
                    mtx_diversity_at_cost += mtx_supply
                    p.update(current_task, advance=1)
                
                # export current cost diversity
                self.write_dataset("INDICATORS/diversity_cost_{}.tif".format(c), mtx_diversity_at_cost) 

        # done
        self.printStepCompleteInfo()


    #
    # Determine flow of beneficiaries to recreational opportunities per cost
    #

    def class_flow(self):
        
        self.printStepInfo("Determine class flow")
        
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch))
        current_task = self.new_progress("[white]Determine class-based flows within cost", step_count)

        with self.progress as p:
            
            for c in self.cost_thresholds:
                mtx_pop = self.read_band("DEMAND/beneficiaries_within_cost_{}.tif".format(c))

                for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):                
                    # determine source of list
                    lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"   
                    mtx_lu = self.get_mask_for_lu(lu, lu_type)
                    mtx_res = mtx_lu * mtx_pop
                    # write result
                    outfile_name = "FLOWS/flow_class_{}_cost_{}.tif".format(lu, c) if lu_type == 'patch' else "FLOWS/flow_edge_class_{}_cost_{}.tif".format(lu, c)
                    self.write_dataset(outfile_name, mtx_res)
                    p.update(current_task, advance=1)

        # done
        self.printStepCompleteInfo()


    #
    #
    # Compute averages in supply, diversity, etc., across cost thresholds.
    # Allow weighting of costs.
    #
    #

    def average_total_supply_across_cost(self, lu_weights = None, cost_weights = None, write_non_weighted_result = True):

        self.printStepInfo("Averaging supply across costs")

        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge))
        current_task = self.new_progress("[white]Averaging supply", step_count)

        # make result rasters
        # consider the following combinations

        # non-weighted lu + non-weighted cost (def. case)
        # non-weighted lu +     weighted cost (computed in addition to def. case if weights supplied)
        #     weighted lu + non-weighted cost (if weights applied only to previous step)
        #     weighted lu +     weighted cost

        with self.progress as p:

            # def. case
            if write_non_weighted_result:
                non_weighted_average_total_supply = self.get_value_matrix()            
            # def. case + cost weighting
            if cost_weights is not None:
                cost_weighted_average_total_supply = self.get_value_matrix()                

            if lu_weights is not None:
                # lu weights only
                lu_weighted_average_total_supply = self.get_value_matrix()
                if cost_weights is not None:
                    # both weights
                    bi_weighted_average_total_supply = self.get_value_matrix()

            # iterate over costs
            for c in self.cost_thresholds:

                # re-aggregate lu supply within cost, using currently supplied weights
                mtx_current_cost_total_supply, mtx_current_cost_weighted_total_supply = self.get_aggregate_class_total_supply_for_cost(c, lu_weights, write_non_weighted_result, current_task)                                           
                
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
                self.write_dataset("INDICATORS/non_weighted_avg_totalsupply.tif", non_weighted_average_total_supply)

            # def. case + cost weighting
            if cost_weights is not None:
                cost_weighted_average_total_supply = cost_weighted_average_total_supply / sum(cost_weights.values())
                self.write_dataset("INDICATORS/cost_weighted_avg_totalsupply.tif", cost_weighted_average_total_supply)
            
            if lu_weights is not None:
                # lu weights only
                lu_weighted_average_total_supply = lu_weighted_average_total_supply / len(self.cost_thresholds)
                self.write_dataset("INDICATORS/landuse_weighted_avg_totalsupply.tif", lu_weighted_average_total_supply)

                if cost_weights is not None:
                    # both weights
                    bi_weighted_average_total_supply = bi_weighted_average_total_supply / sum(cost_weights.values())
                    self.write_dataset("INDICATORS/bi_weighted_avg_totalsupply.tif", bi_weighted_average_total_supply)
            
        # done
        self.printStepCompleteInfo()



    #
    # Average diversity across cost thresholds.
    #

    def average_diversity_across_cost(self, cost_weights = None, write_non_weighted_result = True):

        self.printStepInfo("Averaging diversity across costs")

        step_count = len(self.cost_thresholds)
        current_task = self.new_progress("[white]Averaging diversity", step_count)

        with self.progress as p:

            # result raster
            if write_non_weighted_result:
                average_diversity = self.get_value_matrix()
            if cost_weights is not None:
                cost_weighted_average_diversity = self.get_value_matrix()

            # iterate over cost thresholds and aggregate cost-specific diversities into result
            for c in self.cost_thresholds:
                mtx_current_diversity = self.read_band("INDICATORS/diversity_cost_{}.tif".format(c)) 
                if write_non_weighted_result:
                    average_diversity += mtx_current_diversity
                if cost_weights is not None:
                    cost_weighted_average_diversity += (average_diversity * cost_weights[c])

                p.update(current_task, advance=1)

            # export averaged diversity grids
            if write_non_weighted_result:
                average_diversity = average_diversity / len(self.cost_thresholds)
                self.write_dataset("INDICATORS/non_weighted_avg_diversity.tif", average_diversity)
                        
            if cost_weights is not None:
                cost_weighted_average_diversity = cost_weighted_average_diversity / sum(cost_weights.values())
                self.write_dataset("INDICATORS/cost_weighted_avg_diversity.tif", cost_weighted_average_diversity)
        
        # done
        self.printStepCompleteInfo()

    #
    # Average beneficiaries across cost thresholds.
    #

    def average_beneficiaries_across_cost(self, cost_weights = None, write_non_weighted_result = True):
        
        self.printStepInfo("Averaging beneficiaries across costs")

        step_count = len(self.cost_thresholds)
        current_task = self.new_progress("[white]Averaging beneficiaries", step_count)

        with self.progress as p:

            # result raster
            if write_non_weighted_result:
                average_pop = self.get_value_matrix()
            if cost_weights is not None:
                cost_weighted_average_pop = self.get_value_matrix()

            # iterate over cost thresholds and aggregate cost-specific beneficiaries into result
            for c in self.cost_thresholds:
                mtx_current_pop = self.read_band("DEMAND/beneficiaries_within_cost_{}.tif".format(c)) 
                if write_non_weighted_result:
                    average_pop += mtx_current_pop
                if cost_weights is not None:
                    cost_weighted_average_pop += (mtx_current_pop * cost_weights[c])
                p.update(current_task, advance=1)
            
            # export averaged diversity grids
            if write_non_weighted_result:
                average_pop = average_pop / len(self.cost_thresholds)
                self.write_dataset("INDICATORS/non_weighted_avg_population.tif", average_pop)
            if cost_weights is not None:
                cost_weighted_average_pop = cost_weighted_average_pop / sum(cost_weights.values())
                self.write_dataset("INDICATORS/cost_weighted_avg_population.tif", cost_weighted_average_pop)

        # done
        self.printStepCompleteInfo()



    #
    # Average class flow across cost thresholds
    #
    
    def average_flow_across_cost(self, cost_weights = None, write_non_weighted_result = True):

        self.printStepInfo("Averaging flow across costs")
        
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge)) 
        current_task = self.new_progress("[white]Averaging flow across costs", step_count)

        with self.progress as p:

            # result grids for integrating averaged flows
            if write_non_weighted_result:
                integrated_average_flow = self.get_value_matrix()
            if cost_weights is not None:
                integrated_cost_weighted_average_flow = self.get_value_matrix()

            # iterate over cost thresholds and lu classes                
            for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):               

                # result grids for average flow for current cost threshold
                if write_non_weighted_result:
                    class_average_flow = self.get_value_matrix()
                if cost_weights is not None:
                    cost_weighted_class_average_flow = self.get_value_matrix()

                for c in self.cost_thresholds:
                    # determine source of list
                    lu_type = "patch" if lu in self.lu_classes_recreation_patch else "edge"
                    filename = "FLOWS/flow_class_{}_cost_{}.tif".format(lu, c) if lu_type == 'patch' else "FLOWS/flow_edge_class_{}_cost_{}.tif".format(lu, c)

                    mtx_current_flow = self.read_band(filename) 
                    if write_non_weighted_result:
                        class_average_flow += mtx_current_flow
                    if cost_weights is not None:
                        cost_weighted_class_average_flow += (mtx_current_flow * cost_weights[c])
                    p.update(current_task, advance=1)

                # we have now iterated over cost thresholds
                # export current class-averaged flow, and integrate with final product
                if write_non_weighted_result:
                    class_average_flow = class_average_flow / len(self.cost_thresholds)
                    self.write_dataset("FLOWS/average_flow_class_{}.tif".format(lu), class_average_flow)
                    # add to integrated grid
                    integrated_average_flow += class_average_flow

                if cost_weights is not None:
                    cost_weighted_class_average_flow = cost_weighted_class_average_flow / sum(cost_weights.values())
                    self.write_dataset("FLOWS/cost_weighted_average_flow_class_{}.tif".format(lu), cost_weighted_class_average_flow)
                    # add to integrated grid
                    integrated_cost_weighted_average_flow += cost_weighted_class_average_flow

            # export integrated grids
            if write_non_weighted_result:
                self.write_dataset("FLOWS/integrated_avg_flow.tif", integrated_average_flow)
            if cost_weights is not None:
                self.write_dataset("FLOWS/integrated_cost_weighted_avg_flow.tif", integrated_cost_weighted_average_flow)

        self.printStepCompleteInfo()


    # 
    # Clump detection in land-uses to determine size of patches and edges
    # To determine per-capita recreational area
    #

    def per_capita_opportunity_area(self):
        
        self.printStepInfo("Determining per-capita opportunity area")

        step_count = len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge)
        lu_progress = self.new_progress("[white]Per-capita assessment", step_count)

        # get average flow for all lu patches as basis for average per-capita area
        mtx_average_flows = self.read_band("FLOWS/integrated_avg_flow.tif")
        # make clump raster
        clump_connectivity = np.full((3,3), 1)

        with self.progress as p:

            # particularly for debugging reasons, write out all grids
            res_lu_patch_area_per_capita = self.get_value_matrix()
            res_lu_clump_size = self.get_value_matrix()
            res_lu_clump_average_flow = self.get_value_matrix()

            # iterate over land-uses
            for lu in (self.lu_classes_recreation_patch + self.lu_classes_recreation_edge):
                # determine lu type
                lu_type = 'patch' if lu in self.lu_classes_recreation_patch else 'edge'
                
                # get lu mask as basis to determine clump total area
                mtx_current_lu_mask = self.get_mask_for_lu(lu, lu_type)

                mtx_current_lu_clumps = self.get_value_matrix()
                mtx_current_lu_clump_size = self.get_value_matrix()
                mtx_current_lu_clump_average_flow = self.get_value_matrix()

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
                self.write_dataset('CLUMPS_LU/clump_size_class_{}.tif'.format(lu), mtx_current_lu_clump_size)
                self.write_dataset('CLUMPS_LU/clump_flow_class_{}.tif'.format(lu), mtx_current_lu_clump_average_flow)
                self.write_dataset('CLUMPS_LU/clump_pcap_class_{}.tif'.format(lu), mtx_current_lu_clump_area_per_capita)

                # add to integrated grid
                res_lu_clump_size += mtx_current_lu_clump_size
                res_lu_clump_average_flow += mtx_current_lu_clump_average_flow                
                res_lu_patch_area_per_capita += mtx_current_lu_clump_area_per_capita   

                p.update(lu_progress, advance=1)

        self.write_dataset('CLUMPS_LU/clumps_size.tif', res_lu_clump_size)
        self.write_dataset('CLUMPS_LU/clumps_flow.tif', res_lu_clump_average_flow)
        self.write_dataset('CLUMPS_LU/clumps_pcap.tif', res_lu_patch_area_per_capita)

        # done
        self.printStepCompleteInfo()




    # 
    # Approximation of distance-cost to closest entity per lu class 
    #
    #
    def cost_to_closest(self, upper_threshold = 25):
        self.printStepInfo("Assessing cost to closest")

        included_lu_classes = self.lu_classes_recreation_patch + self.lu_classes_recreation_patch
        step_count = len(included_lu_classes)         
        current_task = self.new_progress("[white]Assessing cost to closest", step_count)

        with self.progress as p:

            with lu in included_lu_classes:
                # import pre-computed proximity raster
                proximity_mtx = self.read_band("PROX/dr_{}.tif".format(lu))
                # mask out values that are greater than upper_threshold as they are considered irrelevant
                # fill higher values with upper threshold
                proximity_mtx[proximity_mtx > upper_threshold] = upper_threshold
                # intersect with built-up to determine closest costs


                p.update(current_task, advance=1)




    #
    # Helper functions
    #
    #

    def read_dataset(self, fileName, band = 1, nodataValues = [0], is_scenario_specific = True):
        path = "{}/{}".format(self.dataPath, fileName) if not is_scenario_specific else "{}/{}/{}".format(self.dataPath, self.scenario_name, fileName)
        if self.verbose_reporting:
            print(Fore.WHITE + Style.DIM + "READING {}".format(path) + Style.RESET_ALL)
        rst_ref = rasterio.open(path)
        band_data = rst_ref.read(band)
        nodata_mask = np.isin(band_data, nodataValues, invert=False)
        return rst_ref, band_data, nodata_mask
        
    def read_band(self, fileName, band = 1, is_scenario_specific = True):
        path = "{}/{}".format(self.dataPath, fileName) if not is_scenario_specific else "{}/{}/{}".format(self.dataPath, self.scenario_name, fileName)
        if self.verbose_reporting:
            print(Fore.WHITE + Style.DIM + "READING {}".format(path) + Style.RESET_ALL)
        rst_ref = rasterio.open(path)
        band_data = rst_ref.read(band)
        return band_data
    
    def write_dataset(self, fileName, outdata, mask_nodata = True, is_scenario_specific = True):        
        path = "{}/{}".format(self.dataPath, fileName) if not is_scenario_specific else "{}/{}/{}".format(self.dataPath, self.scenario_name, fileName)
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
    
    def get_value_matrix(self, fill_value = 0):
        rst_new = np.full(shape=self.lsm_mtx.shape, fill_value=fill_value, dtype=self.lsm_mtx.dtype)
        return rst_new        

    def get_circular_kernel(self, kernel_size):
        kernel = np.zeros((kernel_size,kernel_size))
        radius = kernel_size/2
        # modern scikit uses a tuple for center
        rr, cc = disk( (kernel_size//2, kernel_size//2), radius)
        kernel[rr,cc] = 1
        return kernel

    def kernel_sum(self, subarr):
        return(ndimage.sum(subarr))
    
    def kernel_diversity(self, subarr):
        return len(set(subarr))

    def moving_window(self, data_mtx, kernel_func, kernel_size, kernel_shape = 'circular'):
        # make kernel
        kernel = self.get_circular_kernel(kernel_size) if kernel_shape == 'circular' else np.full((kernel_size, kernel_size), 1)
        # create result mtx as memmap
        mtx_res = np.memmap("{}/{}/TMP/{}".format(self.dataPath, self.scenario_name, uuid.uuid1()), dtype=data_mtx.dtype, mode='w+', shape=data_mtx.shape) 
        # apply moving window over input mtx
        ndimage.generic_filter(data_mtx, kernel_func, footprint=kernel, output=mtx_res, mode='constant', cval=0)
        mtx_res.flush()
        return mtx_res
    
    

    
        
    
    
    

    
    







    

    


