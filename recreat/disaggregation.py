###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################

import numpy as np
import numpy.ma as ma
import rasterio
import rasterio.enums
import os.path
from sklearn.preprocessing import MinMaxScaler
from enum import Enum
from typing import List, Tuple
from string import Template
import uuid

from rich.console import Console
from rich.table import Table

from .transformations import Transformations
from .base import RecreatBase
from .exceptions import DisaggregationError


class DisaggregationMethod(Enum):
    SimpleAreaWeighted = 'saw'
    IntelligentDasymetricMapping = 'idm'

class TransformationState(Enum):
    NotRequired = 'none'
    DownscalePopulation = 'downscale'

class DisaggregationBaseEngine(RecreatBase):

    population_grid: str = None
    residential_classes: List[int] = None
    pixel_count: int = None
    write_scaled_result: bool = True
    transformation_state: TransformationState = None

    nodata_value = -9999.0


    def __init__(self, data_path: str, root_path: str, population_grid: str, residential_classes: List[int], pixel_count: int, nodata_value: any, write_scaled_result: bool = True):
        
        super().__init__(data_path=data_path, root_path=root_path)

        self.population_grid = population_grid
        self.residential_classes = residential_classes
        self.write_scaled_result = write_scaled_result
        self.pixel_count = pixel_count
        self.nodata_value = nodata_value

        if pixel_count == 1:
            # 1-to-1 match between built-up and population
            self.transformation_state = TransformationState.NotRequired
        elif pixel_count > 1:
            # n-to-1 match between built-up and population
            # built-up resolution is higher than population
            self.transformation_state = TransformationState.DownscalePopulation
             

    def get_population_data(self, population_grid: str = None) -> Tuple[rasterio.DatasetReader, np.ndarray]:
        
        # get population
        pop_path = self.get_file_path(population_grid)
        ref_pop = rasterio.open(pop_path)
        mtx_pop = ref_pop.read(1)
        
        # assert correct replacement of nodata values with 0
        mtx_pop[mtx_pop == ref_pop.meta['nodata']] = 0

        return ref_pop, mtx_pop

    def get_file_path(self, file_name: str):
        """Get the fully-qualified path to model file with specified filename.

        :param file_name: Model file for which the fully qualified path should be generated. 
        :type file_name: str 
        """
        return f"{self.data_path}/{self.root_path}/{file_name}"

    def determine_pixel_count_per_population_cell(self, residential_classes: List[int]) -> None:
        """Matches population and built-up rasters and sums built-up pixels within each cell of the population raster. It will write a raster of built-up pixel count.

        :param residential_classes: Residential class values to assess.
        :type residential_class: List[int]
        """        
        step_count = len(residential_classes)
        current_task = self._new_task("[white]Determine pixel count", total=step_count)
        with self.progress:
            for cls in residential_classes:               
                self._determine_pixel_count_per_population_cell(cls)
                self.progress.update(current_task, advance=1)

        # done
        self.taskProgressReportStepCompleted()

    def _determine_pixel_count_per_population_cell(self, residential_class: int) -> None:
       
        # source raster = aggregated builtup area
        # template raster = population
        # target raster = builtup_count        
        out_filename = self.get_file_path(f"DEMAND/pixel_count_{residential_class}.tif") 
        source_filename = self.get_file_path(f"MASKS/mask_{residential_class}.tif")
        template_filename = self.get_file_path(self.population_grid)
        
        if not os.path.isfile(out_filename):
            Transformations.match_rasters(source_filename, template_filename, out_filename, rasterio.enums.Resampling.sum, np.float32)



class DasymetricMappingEngine(DisaggregationBaseEngine):

    samples = None
    sampling_threshold = None
    minimum_sample_size = None

    def __init__(self, data_path: str, root_path: str, population_grid: str, residential_classes: List[int], max_pixel_count: int, count_threshold: int, min_sample_size: int, nodata_value: any, write_scaled_result: bool = True):
        super().__init__(data_path=data_path, root_path=root_path, population_grid=population_grid, residential_classes=residential_classes, pixel_count=max_pixel_count, nodata_value=nodata_value, write_scaled_result=write_scaled_result)

        self.samples = {}
        self.sampling_threshold = count_threshold
        self.minimum_sample_size = min_sample_size

    def run(self):

        self.determine_pixel_count_per_population_cell(self.residential_classes)
        self.sample_classes(self.residential_classes)
        self.match_class_related_sources_to_built_up(self.residential_classes, Template("DEMAND/pixel_count_${residential_class}.tif"), Template("DEMAND/reprojected_pixel_count_${residential_class}.tif"))
        self.match_population_to_built_up()
        self.determine_sum_of_target_zones(self.residential_classes)        
        
        self.disaggregate(self.residential_classes)

    def sample_classes(self, residential_classes: List[int]):
        
        # import data we re-use
        # population 
        ref_pop, mtx_pop = self.get_population_data(self.population_grid)                
        mtx_pop = mtx_pop.flatten()

        for cls in residential_classes:
            self._sample_class_absolute(mtx_pop, cls)
    
        # print a confirmation table
        table = Table(title="Relative class densities")

        table.add_column("Class", justify="left", style="cyan", no_wrap=True)
        table.add_column("Population", justify="right")
        table.add_column("Relative density", justify="right")

        for k,v in self.samples.items():
            table.add_row( f"{k}", f"{v['population']}", f"{v['density']}" )
        
        console = Console()
        console.print(table)

        del mtx_pop

    def _sample_class_absolute(self, mtx_population: np.ndarray, residential_class: int):
        
        # import residential pixel count
        mtx_residential_count = rasterio.open(self.get_file_path(f"DEMAND/pixel_count_{residential_class}.tif")).read(1).flatten()
       
        # mask pixels with count >= sampling threshold
        current_mask = mtx_residential_count < self.sampling_threshold      
        masked_population = ma.array(mtx_population, mask=current_mask)
        masked_pixel_count = ma.array(mtx_residential_count, mask=current_mask)

        # determine if the retrieved pixel count is >= min sample size 
        if masked_pixel_count.count() >= self.minimum_sample_size:
            # this sampling of a class is valid and successful
            # determine total population of this class
            sum_of_pop = np.ma.sum(masked_population)
            cnt_of_pixel = masked_pixel_count.count() 
            sum_of_pixel = np.ma.sum(masked_pixel_count)

            class_density = sum_of_pop/sum_of_pixel
            class_density_from_count = sum_of_pop/cnt_of_pixel
            # insert into class samples
            self.samples[residential_class] = {
                'population' : sum_of_pop,
                'pixel_count' : cnt_of_pixel,
                'pixel_sum' : sum_of_pixel,
                'density' : class_density,
                'count_density' : class_density_from_count
            }

            del mtx_residential_count
            del masked_population
            del masked_pixel_count
            del current_mask

        else:
            # this sampling of a class has failed. 
            raise(DisaggregationError(f"The sampling of class {residential_class} failed with count {masked_pixel_count.count()} < minimum sampling size of {self.minimum_sample_size}.")) 

    def match_class_related_sources_to_built_up(self, residential_classes: List[int], source_template: Template, out_template: Template) -> None:
        """ Matches pixel count and built-up rasters. It will write the reprojected dataset to disk.

        :param residential_classes: Residential class values to assess.
        :type residential_class: List[int]
        """
        step_count = len(residential_classes)
        current_task = self._new_task("[white]Reprojecting raster", total=step_count)
                        
        with self.progress:
            for cls in self.residential_classes:
                source_filename = source_template.safe_substitute(residential_class=cls)
                template_filename = f"MASKS/mask_{cls}.tif"
                out_filename = out_template.safe_substitute(residential_class=cls)

                self._match_source_to_built_up(source_filename, template_filename, out_filename)
                self.progress.update(current_task, advance=1)

        # done
        self.taskProgressReportStepCompleted()

    def match_population_to_built_up(self):
        source_filename = self.population_grid
        template_filename = "BASE/clumps.tif"
        out_filename = "DEMAND/reprojected_population.tif"
        self._match_source_to_built_up(source_filename, template_filename, out_filename)
    
    def _match_source_to_built_up(self, source_filename, template_filename, out_filename, resampling_method: rasterio.enums.Resampling = rasterio.enums.Resampling.min) -> None:

        out_filename = self.get_file_path(out_filename)
        if not os.path.isfile(out_filename):
            source_filename = self.get_file_path(source_filename)
            template_filename = self.get_file_path(template_filename)
            Transformations.match_rasters(source_filename, template_filename, out_filename, resampling_method, np.float32)

    def determine_sum_of_target_zones(self, residential_classes: List[int]) -> None:
        
        out_mtx = None
        dest_meta = None

        for cls in residential_classes:
            class_density = self.samples[cls]['density']
            ref_class_count = rasterio.open(self.get_file_path(f"DEMAND/reprojected_pixel_count_{cls}.tif"))
            mtx_class_count = ref_class_count.read(1)
            
            if out_mtx is None:
                out_mtx = np.zeros(mtx_class_count.shape, dtype=np.float32)
                dest_meta = ref_class_count.meta.copy()
                dest_meta.update({
                    'nodata' : self.nodata_value,
                    'dtype' : np.float32
                })

            out_mtx += np.multiply(class_density, mtx_class_count)
            del mtx_class_count

        # export result
        out_filename = self.get_file_path("DEMAND/sum_atdc.tif")
        with rasterio.open(out_filename, "w", **dest_meta) as dest:
                dest.write(out_mtx, 1)
        self.taskProgressReportStepCompleted()

    def disaggregate_class_population(self, residential_classes: List[int], mtx_pop: np.ndarray) -> None:
        
        step_count = len(residential_classes)
        current_task = self._new_task("[white]Disaggregating class", total=step_count)
                
        mtx_sum_of_targets = rasterio.open(self.get_file_path("DEMAND/sum_atdc.tif")).read(1)       

        with self.progress:            
            for cls in residential_classes:
                
                out_filename = self.get_file_path(f"DEMAND/disaggregated_population_class_{cls}.tif")
                if not os.path.isfile(out_filename):
                    ref_class_count = rasterio.open(self.get_file_path(f"DEMAND/reprojected_pixel_count_{cls}.tif"))
                    mtx_class_count = ref_class_count.read(1)

                    mtx_class_builtup = rasterio.open(self.get_file_path(f"MASKS/mask_{cls}.tif")).read(1)            
                    class_density = self.samples[cls]['density']

                    mtx_out = np.zeros(mtx_class_count.shape, dtype=np.float32)
                    np.divide( (mtx_class_builtup * class_density), mtx_sum_of_targets, out=mtx_out, where=mtx_sum_of_targets > 0)  #* mtx_pop
                    mtx_out = mtx_out * mtx_pop

                    dest_meta = ref_class_count.meta.copy()
                    dest_meta.update({
                        'nodata' : self.nodata_value,
                        'dtype' : np.float32
                    })
                    # export result
                    with rasterio.open(out_filename, "w", **dest_meta) as dest:
                            dest.write(mtx_out, 1)

                    del mtx_out
                    del mtx_class_count
                    del mtx_class_builtup

                self.progress.update(current_task, advance=1)

        del mtx_sum_of_targets

    def disaggregate(self, residential_classes: List[int]) -> None:
        
        # reprojected_population = population of source zone yt
        ref_pop, mtx_pop = self.get_population_data("DEMAND/reprojected_population.tif")
        self.disaggregate_class_population(residential_classes, mtx_pop)
        
        step_count = len(residential_classes)
        current_task = self._new_task("[white]Finalizing", total=step_count)
        mtx_out_population = np.zeros(mtx_pop.shape, dtype=np.float64)

        with self.progress:
            for cls in residential_classes:
                mtx_class_pop = rasterio.open(self.get_file_path(f"DEMAND/disaggregated_population_class_{cls}.tif")).read(1)
                mtx_out_population = np.add(mtx_out_population.astype(np.float64), mtx_class_pop.astype(np.float64))
                self.progress.update(current_task, advance=1)
            
        dest_meta = ref_pop.meta.copy()
        dest_meta.update({
            'nodata' : self.nodata_value,
            'dtype' : np.float64
        })

        # proper masking
        clump_data = rasterio.open(self.get_file_path('BASE/clumps.tif'), 'r').read(1)
        clump_nodata_mask = np.insin(clump_data, [self.nodata_value], invert=False)
        mtx_out_population[clump_nodata_mask] = self.nodata_value

        # export result
        out_filename = self.get_file_path("DEMAND/disaggregated_population.tif")
        with rasterio.open(out_filename, "w", **dest_meta) as dest:
                dest.write(mtx_out_population, 1)

        # TODO: Add scaled export of pop
        


class SimpleAreaWeightedEngine(DisaggregationBaseEngine):

    def __init__(self, data_path, root_path, population_grid: str, residential_classes: List[int], max_pixel_count: int, nodata_value: any, write_scaled_result: bool = True):
        super().__init__(data_path=data_path, root_path=root_path, population_grid=population_grid, residential_classes=residential_classes, pixel_count=max_pixel_count, nodata_value=nodata_value, write_scaled_result=write_scaled_result)

    def run(self) -> None:        
        """Run simple area weighted disaggregation engine. 
        """        
        self.determine_pixel_count_per_population_cell(self.residential_classes)                
        self.determine_class_share(self.residential_classes)        
        self.determine_class_pixel_population(self.residential_classes)
        self.reproject_pixel_population_count_to_builtup(self.residential_classes)
        self.aggregate_class_population(self.residential_classes)

    

    

    def determine_class_share(self, residential_classes: List[int]) -> None:
        """Determine the share of residential class within population raster cell

        :param residential_classes: Residential class values to assess.
        :type residential_class: List[int]
        """
        step_count = len(residential_classes)
        current_task = self._new_task("[white]Determine class pixel share", total=step_count)
        with self.progress:
            for cls in residential_classes:
                self._determine_class_share(cls)
                self.progress.update(current_task, advance=1)
        # done
        self.taskProgressReportStepCompleted()

    def _determine_class_share(self, residential_class: int) -> None:

        out_filename = self.get_file_path(f"DEMAND/class_share{residential_class}.tif")
        if not os.path.isfile(out_filename):

            residential_class_file_path = self.get_file_path(f"DEMAND/pixel_count_{residential_class}.tif")
            ref_residential_pixel_count = rasterio.open(residential_class_file_path)
            
            dest_meta = ref_residential_pixel_count.meta.copy()        
            dest_meta.update({
                'dtype' : rasterio.float64,
                'nodata' : self.nodata_value
            })
                
            mtx_residential_pixel_count = ref_residential_pixel_count.read(1)

            # create result matrix
            mtx_pixel_share = np.zeros(mtx_residential_pixel_count.shape, dtype=np.float64)

            # determine share
            np.divide(mtx_residential_pixel_count.astype(np.float64), self.pixel_count, out=mtx_pixel_share, where=mtx_residential_pixel_count > 0)

            # export result
            with rasterio.open(out_filename, "w", **dest_meta) as dest:
                dest.write(mtx_pixel_share, 1)

            del mtx_pixel_share
            del mtx_residential_pixel_count


    
    def determine_class_pixel_population(self, residential_classes: List[int]) -> None:
        """Applies disaggregation algorithm. At the moment, a simple area-weighted approach is implemented.

        :param residential_classes: Residential class values to assess.
        :type residential_class: List[int]
        """
        step_count = len(residential_classes)
        current_task = self._new_task("[white]Determine per-pixel population", total=step_count)
        with self.progress:
            for cls in residential_classes:
                self._determine_class_pixel_population(cls)
                self.progress.update(current_task, advance=1)
        
        # done
        self.taskProgressReportStepCompleted()

    def _determine_class_pixel_population(self, residential_class: int) -> None:
    
        out_filename = self.get_file_path(f"DEMAND/pixel_population_count_{residential_class}.tif")
        if not os.path.isfile(out_filename):

            ref_pop, mtx_pop = self.get_population_data(self.population_grid)
            dest_meta = ref_pop.meta.copy()
            dest_meta.update({
                'dtype' : rasterio.float32,
                'nodata' : self.nodata_value
            })

            # target matrix
            mtx_per_pixel_population_count = np.zeros(mtx_pop.shape, dtype=np.float32)
            
            # get share matrix
            mtx_share_path = self.get_file_path(f"DEMAND/class_share{residential_class}.tif")
            mtx_share = rasterio.open(mtx_share_path).read(1)

            # make sure that a float dtype is set       
            # first, multiply share of class with population, to determine actual pop to be distributed        
            np.multiply(mtx_share.astype(np.float32), mtx_pop.astype(np.float32), out=mtx_per_pixel_population_count)   
            del mtx_share
            del mtx_pop


            # second, divide population to be disaggregated by pixel count to determine per-pixel populationy figure
            # get class pixel count per population grid cell
            pixel_count_path = self.get_file_path(f"DEMAND/pixel_count_{residential_class}.tif")
            mtx_pixel_count = rasterio.open(pixel_count_path).read(1)        
            np.divide(mtx_per_pixel_population_count.astype(np.float32), mtx_pixel_count.astype(np.float32), out=mtx_per_pixel_population_count, where=mtx_pixel_count > 0)        

            # export result
            with rasterio.open(out_filename, "w", **dest_meta) as dest:
                dest.write(mtx_per_pixel_population_count.astype(np.float32), 1)
            
            del mtx_pixel_count
            del mtx_per_pixel_population_count   

    def reproject_pixel_population_count_to_builtup(self, residential_classes: List[int]) -> None:
        """ Matches patch population and built-up rasters. It will write the reprojected dataset to disk.

        :param residential_classes: Residential class values to assess.
        :type residential_class: List[int]
        """
        step_count = len(residential_classes)
        current_task = self._new_task("[white]Determine class population", total=step_count)
        with self.progress:
            for cls in self.residential_classes:
                self._reproject_pixel_population_count_to_builtup(cls)
                self.progress.update(current_task, advance=1)

        # done
        self.taskProgressReportStepCompleted()


    def _reproject_pixel_population_count_to_builtup(self, residential_class: int) -> None:

        # source raster = patch_population
        # template raster = built-up area
        # target raster = reprojected_patch_population 
        out_filename = self.get_file_path(f"DEMAND/population_base_{residential_class}.tif")
        if not os.path.isfile(out_filename):
            source_filename = self.get_file_path(f"DEMAND/pixel_population_count_{residential_class}.tif")
            template_filename = self.get_file_path(f"MASKS/mask_{residential_class}.tif")
            Transformations.match_rasters(source_filename, template_filename, out_filename, rasterio.enums.Resampling.min, np.float32)

    def aggregate_class_population(self, residential_classes: List[int]) -> None:
        step_count = 1
        current_task = self._new_task("[white]Determine class population", total=step_count)
        with self.progress:
            self._aggregate_class_population(residential_classes)
            self.progress.update(current_task, advance=1)

        # done
        self.taskProgressReportStepCompleted()

    def _aggregate_class_population(self, residential_classes: List[int], current_task = None) -> None:

        out_filename = self.get_file_path('DEMAND/disaggregated_population.tif')
        if not os.path.isfile(out_filename):

            dest_mtx = None
            dest_meta = None
            
            for cls in residential_classes:
                
                # add pop to final raster
                class_mask_path = self.get_file_path(f"MASKS/mask_{cls}.tif")
                ref_class_mask = rasterio.open(class_mask_path)
                mtx_class_mask = ref_class_mask.read(1)
                if dest_mtx is None:
                    
                    # import raster reference
                    dest_mtx = np.zeros(ref_class_mask.shape, dtype=np.float64)
                    dest_meta = ref_class_mask.meta.copy()
                    dest_meta.update({
                        'dtype' : rasterio.float64,
                        'nodata' : self.nodata_value
                    })

                class_pixel_population_path = self.get_file_path(f"DEMAND/population_base_{cls}.tif")
                ref_class_pixel_population = rasterio.open(class_pixel_population_path)
                mtx_class_pixel_population = ref_class_pixel_population.read(1)
                
                # intersect mask and per-pixel population count
                mtx_class_mask = mtx_class_mask * mtx_class_pixel_population
                
                # add to dest_mtx
                dest_mtx += mtx_class_mask

                # some clean-up
                del mtx_class_mask
                del mtx_class_pixel_population

                # update progress
                if current_task is not None:
                    self.progress.update(current_task, advance=1)

            # proper masking by clumps
            clump_data = rasterio.open(self.get_file_path('BASE/clumps.tif'), 'r').read(1)
            clump_nodata_mask = np.isin(clump_data, [self.nodata_value], invert=False)
            dest_mtx[clump_nodata_mask] = self.nodata_value

            with rasterio.open(out_filename, "w", **dest_meta) as dest:
                dest.write(dest_mtx, 1)
            
            # if self.write_scaled_result:
            #     scaler = MinMaxScaler()
            #     orig_shape = dest_mtx.shape
            #     dest_mtx = scaler.fit_transform(dest_mtx.reshape([-1,1]))
            #     out_filename = self.get_file_path('DEMAND/scaled_disaggregated_population.tif')
                
            #     with rasterio.open(out_filename, "w", **dest_meta) as dest:
            #         dest.write(dest_mtx.reshape(orig_shape), 1)

        