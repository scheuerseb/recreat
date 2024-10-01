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

colorama_init()

class LsmGridRecreationModeller:

    dataPath = None
    pop_fileName = None
    lsm_fileName = None
    lsm_rst = None
    lsm_mtx = None
    lsm_nodataMask = None

    lu_classes_recreation_edge = []
    lu_classes_recreation_patch = []
    lu_classes_builtup = []
    cost_thresholds = []
    clump_slices = []

    
    verbose_reporting = False

    # progress reporting
    progress = None 
    
    task_overall = None

    def __init__(self, dataPath, lsmFileName, populationFileName):
        os.system('cls' if os.name == 'nt' else 'clear')
        self.dataPath = dataPath
        self.lsm_fileName = lsmFileName
        self.pop_fileName = populationFileName        
        self.make_environment()         
        self.progress = self.getProgressBar()       
    
    def make_environment(self):
        # create directories, if needed
        dirs_required = ['DEMAND', 'MASKS', 'SUPPLY', 'INDICATORS', 'TMP', 'FLOWS']
        for d in dirs_required:
            cpath = "{}/{}".format(self.dataPath, d)
            if not os.path.exists(cpath):
                os.makedirs(cpath)

    def printStepInfo(self, msg):
        print(Fore.CYAN + Style.DIM + msg.upper() + Style.RESET_ALL)

    def getProgressBar(self):
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

    def assess_map_units(self):       
        
        with self.progress as p:
            
            # import raster
            self.task_overall = self.progress.add_task('[red]Assessing recreational potential', total=6)
            self.lsm_rst, self.lsm_mtx, self.lsm_nodataMask = self.read_dataset(self.lsm_fileName)
            
            # detecting clumps
            self.detect_clumps()
                        
            # mask land-uses and detect edges on relevant classes
            self.mask_landuses()            
            self.detect_edges()

            # work on population disaggregation
            self.disaggregate_population()            
            self.beneficiaries_within_cost()

            # determine supply per class
            self.class_total_supply()

        print("DONE ASSESSING MAP UNITS. UBER WISHES YOU A JOLLY PLEASANT DAY.")



    def advanceStepTotal(self):
        self.progress.update(self.task_overall, advance=1)




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

    def class_total_supply_for_lu_and_cost(self, mask_path, rst_clumps, cost, progress_bar = None):
        
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

            if progress_bar is not None:
                self.progress.update(progress_bar, advance=1)
        
        # done with current iterations. return result
        del full_lu_mtx
        return lu_supply_mtx
    


    #
    # The following class aggregates lu-class-specific supply within a given cost to total supply within cost.
    # It can be called following assessment of map units.
    # A weighting schema for lu classes can be supplied.
    #
    
    def aggregate_class_total_supply(self, lu_weights = None, outfile_template = None, write_non_weighted_result = True):

        self.printStepInfo('Determining clumped total supply')
        
        # progress reporting
        self.progress = Progress()
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge))        
        task_aggr = self.progress.add_task("[white]Aggregating clumped supply", total=step_count)

        if outfile_template is None:
            outfile_template = "totalsupply_cost"

        with self.progress as p:

            for c in self.cost_thresholds:

                # make grids for the results: zero-valued grids with full lsm extent
                if write_non_weighted_result:
                    current_total_supply_at_cost = self.get_value_matrix() 
                if lu_weights is not None:
                    current_weighted_total_supply_at_cost = self.get_value_matrix()
            
                for lu in self.lu_classes_recreation_patch:
                    # get supply of current class 
                    lu_supply_mtx = self.read_band("SUPPLY/totalsupply_class_{}_cost_{}_clumped.tif".format(lu, c))                
                    
                    if write_non_weighted_result:
                        current_total_supply_at_cost += lu_supply_mtx
                    if lu_weights is not None:
                        current_weighted_total_supply_at_cost += (lu_supply_mtx * lu_weights[lu]) 
                    
                    self.progress.update(task_aggr, advance=1)

                for lu in self.lu_classes_recreation_edge:
                    # get supply of current class 
                    lu_supply_mtx = self.read_band("SUPPLY/totalsupply_edge_class_{}_cost_{}_clumped.tif".format(lu, c))  
                    
                    if write_non_weighted_result:
                        current_total_supply_at_cost += lu_supply_mtx
                    if lu_weights is not None:
                        current_weighted_total_supply_at_cost += (lu_supply_mtx * lu_weights[lu]) 
                    
                    self.progress.update(task_aggr, advance=1)

                # export total for costs, if requested
                if write_non_weighted_result:                
                    self.write_dataset("SUPPLY/{}_{}.tif".format(outfile_template, c), current_total_supply_at_cost)
                
                # export weighted total, if aplicable
                if lu_weights is not None:
                    current_weighted_total_supply_at_cost = current_weighted_total_supply_at_cost / sum(lu_weights.values())
                    self.write_dataset("SUPPLY/weighted_{}_{}.tif".format(outfile_template, c), current_weighted_total_supply_at_cost)

        # done
        print("Done.")
        








    def class_diversity(self):
        self.printStepInfo("Determining class diversity within costs")        
        step_count = (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch)) * len(self.cost_thresholds)        
        task_classdiv = self.progress.add_task("[white]Determining class diversity", total=step_count)

        for c in self.cost_thresholds:            
            mtx_diversity_at_cost = self.get_value_matrix()
            for lu in self.lu_classes_recreation_edge:
                mtx_supply = self.read_band("SUPPLY/totalsupply_edge_class_{}_cost_{}_clumped.tif".format(lu, c))                
                mtx_supply[mtx_supply > 0] = 1
                mtx_diversity_at_cost += mtx_supply
                self.progress.update(task_classdiv, advance=1)
            
            for lu in self.lu_classes_recreation_patch:
                mtx_supply = self.read_band("SUPPLY/totalsupply_class_{}_cost_{}_clumped.tif".format(lu, c))                
                mtx_supply[mtx_supply > 0] = 1
                mtx_diversity_at_cost += mtx_supply
                self.progress.update(task_classdiv, advance=1)

            # export current cost diversity
            self.write_dataset("INDICATORS/diversity_cost_{}.tif".format(c), mtx_diversity_at_cost) 

        # done
        self.advanceStepTotal()


    def read_dataset(self, fileName, band = 1, nodataValues = [0]):
        rstPath = "{}/{}".format(self.dataPath, fileName)
        if self.verbose_reporting:
            print(Fore.WHITE + Style.DIM + "READING {}".format(rstPath) + Style.RESET_ALL)
        rst_ref = rasterio.open(rstPath)
        band_data = rst_ref.read(band)
        nodata_mask = np.isin(band_data, nodataValues, invert=False)
        return rst_ref, band_data, nodata_mask
    
    
    def read_band(self, fileName, band = 1):
        rstPath = "{}/{}".format(self.dataPath, fileName)
        if self.verbose_reporting:
            print(Fore.WHITE + Style.DIM + "READING {}".format(rstPath) + Style.RESET_ALL)
        rst_ref = rasterio.open(rstPath)
        band_data = rst_ref.read(band)
        return band_data
    
    def write_dataset(self, fileName, outdata, mask_nodata = True):        
        path = "{}/{}".format(self.dataPath, fileName)
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
        mtx_res = np.memmap("{}/TMP/{}".format(self.dataPath, uuid.uuid1()), dtype=data_mtx.dtype, mode='w+', shape=data_mtx.shape) 
        # apply moving window over input mtx
        ndimage.generic_filter(data_mtx, kernel_func, footprint=kernel, output=mtx_res, mode='constant', cval=0)
        mtx_res.flush()
        return mtx_res
    
    

    
        
    
    
    

    
    def class_flow(self):
        print("DETERMINING CLASS FLOW")
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_edge) + len(self.lu_classes_recreation_patch))
        
        with alive_bar(step_count) as bar:
            for c in self.cost_thresholds:
                mtx_pop = self.read_band("DEMAND/beneficiaries_within_cost_{}.tif".format(c))

                for lu in self.lu_classes_recreation_patch:
                    mtx_lu = self.read_band("MASKS/mask_{}.tif".format(lu))
                    mtx_res = mtx_lu * mtx_pop
                    self.write_dataset("FLOWS/flow_class_{}_cost{}.tif".format(lu, c), mtx_res)
                    bar()

                for lu in self.lu_classes_recreation_edge:
                    mtx_lu = self.read_band("MASKS/edges_{}.tif".format(lu))
                    mtx_res = mtx_lu * mtx_pop
                    self.write_dataset("FLOWS/flow_edge_class_{}_cost{}.tif".format(lu, c), mtx_res)
                    bar()







    def average_total_supply_across_cost(self):

        self.printStepInfo("Averaging supply across cost")
        step_count = len(self.cost_thresholds) + 3
        task_avg = self.progress.add_task("[white]Averaging supply", total=step_count)

        # make result rasters
        # whereas for landuse aggregation, there were two grids possible (non-weighted/weighted)
        # here we have more combinations.

        # non-weighted lu + non-weighted cost (def. case)
        # non-weighted lu +     weighted cost (computed in addition to def. case if weights supplied)
        #     weighted lu + non-weighted cost (if weights applied only to previous step)
        #     weighted lu +     weighted cost

        # def. case
        non_weighted_average_total_supply = self.get_value_matrix()            
        # def. case + cost weighting
        if self.apply_cost_weighting():
            cost_weighted_average_total_supply = self.get_value_matrix()                

        if self.apply_landuse_weighting():
            # lu weights only
            lu_weighted_average_total_supply = self.get_value_matrix()
            if self.apply_cost_weighting():
                # both weights
                bi_weighted_average_total_supply = self.get_value_matrix()

        # iterate over costs
        for c in self.cost_thresholds:
            mtx_current_cost_total_supply = self.read_band("SUPPLY/totalsupply_cost_{}.tif".format(c))
            
            non_weighted_average_total_supply += mtx_current_cost_total_supply
            if self.apply_cost_weighting():
                cost_weighted_average_total_supply += (mtx_current_cost_total_supply * self.weights_cost[c])
        
            if self.apply_landuse_weighting():
                # lu weights only
                # get respective dataset
                mtx_current_cost_weighted_total_supply = self.read_band("SUPPLY/weighted_totalsupply_cost_{}.tif".format(c))
                lu_weighted_average_total_supply += mtx_current_cost_weighted_total_supply
                
                if self.apply_cost_weighting():
                    # both weights
                    bi_weighted_average_total_supply += (mtx_current_cost_weighted_total_supply * self.weights_cost[c])

            self.progress.update(task_avg, advance=1)

        # complete determining averages
        # def. case
        non_weighted_average_total_supply = non_weighted_average_total_supply / len(self.cost_thresholds)
        self.write_dataset("INDICATORS/non_weighted_avg_totalsupply.tif", non_weighted_average_total_supply)
        self.progress.update(task_avg, advance=1)


        # def. case + cost weighting
        if self.apply_cost_weighting():
            cost_weighted_average_total_supply = cost_weighted_average_total_supply / sum(self.weights_cost.values())
            self.write_dataset("INDICATORS/cost_weighted_avg_totalsupply.tif", cost_weighted_average_total_supply)
        self.progress.update(task_avg, advance=1)
        

        if self.apply_landuse_weighting():
            # lu weights only
            lu_weighted_average_total_supply = lu_weighted_average_total_supply / len(self.cost_thresholds)
            self.write_dataset("INDICATORS/landuse_weighted_avg_totalsupply.tif", lu_weighted_average_total_supply)

            if self.apply_cost_weighting():
                # both weights
                bi_weighted_average_total_supply = bi_weighted_average_total_supply / sum(self.weights_cost.values())
                self.write_dataset("INDICATORS/bi_weighted_avg_totalsupply.tif", bi_weighted_average_total_supply)
        self.progress.update(task_avg, advance=1)
        
        # done
        self.advanceStepTotal()

    def average_diversity_across_cost(self):

        self.printStepInfo("Averaging diversity")
        step_count = len(self.cost_thresholds) + 2
        task_avg = self.progress.add_task("[white]Averaging diversity", total=step_count)

        average_diversity = self.get_value_matrix()
        if self.apply_cost_weighting():
            cost_weighted_average_diversity = self.get_value_matrix()

        for c in self.cost_thresholds:
            mtx_current_diversity = self.read_band("INDICATORS/diversity_cost_{}.tif".format(c)) 
            average_diversity += mtx_current_diversity
            if self.apply_cost_weighting():
                cost_weighted_average_diversity += (average_diversity * self.weights_cost[c])

            self.progress.update(task_avg, advance=1)


        # export averaged diversity grids
        average_diversity = average_diversity / len(self.cost_thresholds)
        self.write_dataset("INDICATORS/non_weighted_avg_diversity.tif", average_diversity)
        self.progress.update(task_avg, advance=1)
        
        
        if self.apply_cost_weighting():
            cost_weighted_average_diversity = cost_weighted_average_diversity / sum(self.weights_cost.values())
            self.write_dataset("INDICATORS/cost_weighted_avg_diversity.tif", cost_weighted_average_diversity)
        self.progress.update(task_avg, advance=1)
        
        # done
        self.advanceStepTotal()

    def average_beneficiaries_across_cost(self):
        
        self.printStepInfo("Averaging beneficiaries")
        step_count = len(self.cost_thresholds) + 2
        task_avg = self.progress.add_task("[white]Averaging beneficiaries", total=step_count)

        average_pop = self.get_value_matrix()
        if self.apply_cost_weighting():
            cost_weighted_average_pop = self.get_value_matrix()

        for c in self.cost_thresholds:
            mtx_current_pop = self.read_band("DEMAND/beneficiaries_within_cost_{}.tif".format(c)) 
            average_pop += mtx_current_pop
            if self.apply_cost_weighting():
                cost_weighted_average_pop += (mtx_current_pop * self.weights_cost[c])
            self.progress.update(task_avg, advance=1)
        
        # export averaged diversity grids
        average_pop = average_pop / len(self.cost_thresholds)
        self.write_dataset("INDICATORS/non_weighted_avg_population.tif", average_pop)
        self.progress.update(task_avg, advance=1)
        if self.apply_cost_weighting():
            cost_weighted_average_pop = cost_weighted_average_pop / sum(self.weights_cost.values())
            self.write_dataset("INDICATORS/cost_weighted_avg_population.tif", cost_weighted_average_pop)
        self.progress.update(task_avg, advance=1)

        # done
        self.advanceStepTotal()


    def average_flow_across_cost(self):

        print("DETERMINING AVERAGE CLASS FLOW ACROSS COST")
        step_count = len(self.cost_thresholds) * (len(self.lu_classes_recreation_patch) + len(self.lu_classes_recreation_edge)) 

        with alive_bar(step_count) as bar:

            average_flow = self.get_value_matrix()
            if self.apply_cost_weighting():
                cost_weighted_average_flow = self.get_value_matrix()

            # iterate over lu patch classes
            for lu in self.lu_classes_recreation_patch:
                average_class_flow = self.get_value_matrix()
                if self.apply_cost_weighting():
                    cost_weighted_average_class_flow = self.get_value_matrix() 

                for c in self.cost_thresholds:
                    mtx_current_flow = self.read_band("FLOWS/flow_class_{}_cost{}.tif".format(lu, c)) 
                    average_flow += mtx_current_flow
                    average_class_flow += mtx_current_flow

                    if self.apply_cost_weighting():
                        cost_weighted_average_flow += (mtx_current_flow * self.weights_cost[c])
                        cost_weighted_average_class_flow += (mtx_current_flow * self.weights_cost[c])
                    bar()

                # export class flow            
                average_class_flow = average_class_flow / len(self.cost_thresholds)
                self.write_dataset("FLOWS/average_flow_class_{}.tif".format(lu), average_class_flow)
                if self.apply_cost_weighting():
                    cost_weighted_average_class_flow = cost_weighted_average_class_flow / sum(self.weights_cost.values())
                    self.write_dataset("FLOWS/cost_weighted_average_flow_class_{}.tif".format(lu), cost_weighted_average_class_flow)


            # iterate over lu edge classes

            
            # here, the weights for average cost-weighted flow are different, as they are lu times the sum of cost values 
            
            
            # export averaged diversity grids
            #average_flow = average_flow / len(self.cost_thresholds)
            #self.write_dataset("INDICATORS/non_weighted_avg_population.tif", average_flow)
            
            #if self.apply_cost_weighting():
            #    cost_weighted_average_flow = cost_weighted_average_flow / sum(self.weights_cost.values())
            #    self.write_dataset("INDICATORS/cost_weighted_avg_population.tif", cost_weighted_average_flow)
