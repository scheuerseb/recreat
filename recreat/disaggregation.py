import numpy as np
import numpy.ma as ma
import rasterio
import rasterio.enums
from sklearn.preprocessing import MinMaxScaler
from enum import Enum
from typing import List

from rich.progress import Progress, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn, BarColumn

from .transformations import Transformations
import os.path


class DisaggregationMethod(Enum):
    SimpleAreaWeighted = 'saw'
    DasymetricMapping = 'idm'

class TransformationState(Enum):
    NotRequired = 'none'
    DownscalePopulation = 'downscale'

class BaseDisaggregation:

    root_path: str = None
    data_path: str = None
    population_grid: str = None
    residential_classes: List[int] = None
    pixel_count: int = None
    write_scaled_result: bool = True
    transformation_state: TransformationState = None
    progress = None


    def __init__(self, data_path: str, root_path: str, population_grid: str, residential_classes: List[int], pixel_count: int, write_scaled_result: bool = True):
        self.root_path = root_path
        self.data_path = data_path
        self.population_grid = population_grid
        self.residential_classes = residential_classes
        self.write_scaled_result = write_scaled_result


        if pixel_count == 1:
            # 1-to-1 match between built-up and population
            self.transformation_state = TransformationState.NotRequired
        elif pixel_count > 1:
            # n-to-1 match between built-up and population
            # built-up resolution is higher than population
            self.transformation_state = TransformationState.DownscalePopulation
             
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn()
        ) 

    def get_file_path(self, file_name: str):
        """Get the fully-qualified path to model file with specified filename.

        :param file_name: Model file for which the fully qualified path should be generated. 
        :type file_name: str 
        """
        return f"{self.data_path}/{self.root_path}/{file_name}"


class DasymetricMapping(BaseDisaggregation):

    samples = None

    def __init__(self, data_path, root_path, population_grid: str, residential_classes: List[int], write_scaled_result: bool = True):
        super().__init__(data_path=data_path, root_path=root_path, population_grid=population_grid, residential_classes=residential_classes, write_scaled_result=write_scaled_result)

        self.samples = {}

    def sample_class(self, rst_residential_count: np.ndarray, rst_population: np.ndarray, threshold_count: int, min_sample_size: int = 3):                

        mask_residential = rst_residential_count >= threshold_count
        
        masked_residential = ma.array(rst_residential_count, mask=mask_residential)
        masked_pop = ma.array(rst_population, mask=mask_residential)

        print(np.sum(rst_residential_count))
        print(np.sum(rst_population))

        print('masked')
        print(np.ma.sum(masked_residential))
        print(np.ma.sum(masked_pop))


class SimpleAreaWeighted(BaseDisaggregation):

    def __init__(self, data_path, root_path, population_grid: str, residential_classes: List[int], max_pixel_count: int, write_scaled_result: bool = True):
        super().__init__(data_path=data_path, root_path=root_path, population_grid=population_grid, residential_classes=residential_classes, pixel_count=max_pixel_count, write_scaled_result=write_scaled_result)
        self.pixel_count = max_pixel_count

    def run(self) -> None:        
        """Run simple area weighted disaggregation engine. 
        """
        self.determine_pixel_count_per_population_cell(self.residential_classes)                
        self.determine_class_share(self.residential_classes)        
        self.determine_class_pixel_population(self.residential_classes)
        self.reproject_pixel_population_count_to_builtup(self.residential_classes)
        self.aggregate_class_population(self.residential_classes)

    def determine_pixel_count_per_population_cell(self, residential_classes: List[int]) -> None:
        """Matches population and built-up rasters and sums built-up pixels within each cell of the population raster. It will write a raster of built-up pixel count.

        :param residential_classes: Residential class values to assess.
        :type residential_class: List[int]
        """        
        step_count = len(residential_classes)
        current_task = self.progress.add_task("Determine pixel count", total=step_count)
        with self.progress:
            for cls in residential_classes:
                self._determine_pixel_count_per_population_cell(cls)
                self.progress.update(current_task, advance=1)


    def _determine_pixel_count_per_population_cell(self, residential_class: int) -> None:
       
        # source raster = aggregated builtup area
        # template raster = population
        # target raster = builtup_count
        out_filename = self.get_file_path(f"DEMAND/pixel_count_{residential_class}.tif") 
        if not os.path.isfile(out_filename):
            source_filename = self.get_file_path(f"MASKS/mask_{residential_class}.tif")
            template_filename = self.get_file_path(self.population_grid)
            Transformations.match_rasters(source_filename, template_filename, out_filename, rasterio.enums.Resampling.sum, np.float32)

    def determine_class_share(self, residential_classes: List[int]) -> None:
        """Determine the share of residential class within population raster cell

        :param residential_classes: Residential class values to assess.
        :type residential_class: List[int]
        """
        step_count = len(residential_classes)
        current_task = self.progress.add_task("Determine class pixel share", total=step_count)
        with self.progress:
            for cls in residential_classes:
                self._determine_class_share(cls)
                self.progress.update(current_task, advance=1)

    def _determine_class_share(self, residential_class: int) -> None:

        out_filename = self.get_file_path(f"DEMAND/class_share{residential_class}.tif")
        if not os.path.isfile(out_filename):

            residential_class_file_path = self.get_file_path(f"DEMAND/pixel_count_{residential_class}.tif")
            ref_residential_pixel_count = rasterio.open(residential_class_file_path)
            
            dest_meta = ref_residential_pixel_count.meta.copy()        
            dest_meta.update({
                'dtype' : rasterio.float64,
                'nodata' : -127.0
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
        current_task = self.progress.add_task("Determine per-pixel population", total=step_count)
        with self.progress:
            for cls in residential_classes:
                self._determine_class_pixel_population(cls)
                self.progress.update(current_task, advance=1)

    def _determine_class_pixel_population(self, residential_class: int) -> None:
    
        out_filename = self.get_file_path(f"DEMAND/pixel_population_count_{residential_class}.tif")
        if not os.path.isfile(out_filename):

            # get population
            pop_path = self.get_file_path(self.population_grid)
            ref_pop = rasterio.open(pop_path)
            mtx_pop = ref_pop.read(1)
            # assert correct replacement of nodata values with 0
            print(ref_pop.meta)
            mtx_pop[mtx_pop == ref_pop.meta['nodata']] = 0
            
            dest_meta = ref_pop.meta.copy()
            dest_meta.update({
                'dtype' : rasterio.float32,
                'nodata' : -127.0
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
        current_task = self.progress.add_task("Determine class population", total=step_count)
        with self.progress:
            for cls in self.residential_classes:
                self._reproject_pixel_population_count_to_builtup(cls)
                self.progress.update(current_task, advance=1)


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
        current_task = self.progress.add_task("Determine class population", total=step_count)
        with self.progress:
            self._aggregate_class_population(residential_classes)
            self.progress.update(current_task, advance=1)



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
                    dest_mtx = np.zeros(ref_class_mask.shape, dtype=np.float32)
                    dest_meta = ref_class_mask.meta.copy()
                    dest_meta.update({
                        'dtype' : rasterio.float32
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


            with rasterio.open(out_filename, "w", **dest_meta) as dest:
                dest.write(dest_mtx, 1)
            
            if self.write_scaled_result:
                scaler = MinMaxScaler()
                orig_shape = dest_mtx.shape
                dest_mtx = scaler.fit_transform(dest_mtx.reshape([-1,1]))
                out_filename = self.get_file_path('DEMAND/scaled_disaggregated_population.tif')
                
                with rasterio.open(out_filename, "w", **dest_meta) as dest:
                    dest.write(dest_mtx.reshape(orig_shape), 1)

        