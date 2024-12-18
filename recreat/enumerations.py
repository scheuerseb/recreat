###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################

from enum import Enum


class RecreatBaseEnum(Enum):
    def label(self):
        return self.value[0]
    def name(self):
        return self.value[1]

class ClassType(RecreatBaseEnum):
    """Enumeration of model class types"""
    Edge = 'edge classes', 'classes.edge'
    BufferedEdge = 'buffered-edge classes', 'buffered-edge'
    Patch = 'patch classes', 'classes.patch'
    Built_up = 'built-up classes', 'classes.builtup' 

class ModelParameter(RecreatBaseEnum):
    DataType = 'data type', 'use-data-type'
    Verbosity = 'verbose', 'verbose-reporting'
    Costs = 'costs', 'costs'

class ModelEnvironment(RecreatBaseEnum):
    DataPath = 'data-path', 'data_path'
    Scenario = 'land-use raster file', 'land_use_data'
    clean_temporary_files = 'clean-temp-path'

class ScenarioParameters(RecreatBaseEnum):
    RootPath = 'root-path', 'root_path'
    LanduseFileName = 'land-use raster', 'land_use_filename'
    NodataValues = 'nodata values', 'nodata_values'
    NodataFillValue = 'fill value', 'nodata_fill_value'


class CoreTask(RecreatBaseEnum):
    Reclassification = 'reclassification', 'reclassify'
    ClumpDetection = 'clumps', 'detect_clumps'
    MaskLandUses = 'mask-landuses', 'mask_landuses'
    EdgeDetection = 'detect-edges', 'detect_edges'
    ClassTotalSupply = 'class-total-supply', 'class_total_supply'
    AggregateTotalSupply = 'aggregate-total-supply', 'aggregate_class_total_supply'
    AverageTotalSupplyAcrossCost = 'average-total-supply', 'average_total_supply_across_cost'
    ClassDiversity = 'class-diversity', 'class_diversity'
    AverageDiversityAcrossCost = 'average-diversity', 'average_diversity_across_cost'
    Disaggregation = 'disaggregate-population', 'disaggregation'
    ClassFlow = 'class-flow', 'class_flow'
    ComputeDistanceRasters = 'proximities', 'compute_distance_rasters'
    CostToClosest = 'average-cost', 'cost_to_closest'
    
class ClusteringTask(RecreatBaseEnum):
    kmeans = 'k-means clustering', 'kmeans'

    
class ParameterNames:

    class Reclassification(RecreatBaseEnum):
        Mappings = 'mappings', 'mappings'
        ExportFilename = 'export', 'export_filename'
    
    class ClumpDetection(RecreatBaseEnum):
        BarrierClasses = 'barriers', 'barrier_classes'
    
    class EdgeDetection(RecreatBaseEnum):
        LandUseClasses = 'classes', 'lu_classes'
        IgnoreEdgesToClass = 'ignore', 'ignore_edges_to_class'
        BufferEdges = 'buffer', 'buffer_edges'
    
    class ClassTotalSupply(RecreatBaseEnum):
        Mode = 'mode', 'mode'

    class AggregateClassTotalSupply(RecreatBaseEnum):
        LandUseWeights = 'land-use weights', 'lu_weights'
        WriteNonWeightedResult = 'write non-weighted results', 'write_non_weighted_result'

    class AverageTotalSupplyAcrossCost(RecreatBaseEnum):
        LandUseWeights = 'land-use weights', 'lu_weights'
        CostWeights = 'cost weights', 'cost_weights'
        WriteNonWeightedResult = 'write non-weighted results', 'write_non_weighted_result'
        WriteScaledResult = 'write scaled results', 'write_scaled_result'

    class AverageDiversityAcrossCost(RecreatBaseEnum):
        CostWeights = 'cost weights', 'cost_weights'
        WriteNonWeightedResult =  'write non-weighted results', 'write_non_weighted_result'
        WriteScaledResult = 'write scaled results', 'write_scaled_result'

    class ComputeDistanceRasters(RecreatBaseEnum):
        Mode = 'mode', 'mode'
        LandUseClasses = 'land-use classes', 'lu_classes'
        AssessBuiltUp = 'assess proximity to built-up', 'assess_builtup' 

    class CostToClosest(RecreatBaseEnum):
        DistanceThreshold = 'threshold', 'distance_threshold'
        MaskBuiltUp = 'mask built-up', 'builtup_masking'
        WriteScaledResult = 'write scaled results', 'write_scaled_result'

    class Disaggregation(RecreatBaseEnum):
        PopulationRaster = 'population raster filename', 'population_grid'
        DisaggregationMethod = 'disaggregation method', 'disaggregation_method'
        ResidentialPixelToPopulationPixelCount = 'pixel count', 'max_pixel_count'
        WriteScaledResult = 'write scaled results', 'write_scaled_result'
        MinimumSampleSize = 'sample size', 'min_sample_size'
        ClassSampleThreshold = 'sample threshold', 'count_threshold' 
      