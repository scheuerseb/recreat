###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################


from .model import Model, ModelParameter, ModelEnvironment, ScenarioParameters, CoreTask, ClusteringTask, ClassType

from .Configuration import Configuration
from .enumerations import ParameterNames
from .exceptions import ModelValidationError
from .disaggregation import DisaggregationMethod

import click
import pathlib

from colorama import Fore, Style, just_fix_windows_console
just_fix_windows_console()




cli_model = Model()


@click.group(invoke_without_command=True, chain=True)
@click.option('-w', '--data-path', default=None, help="Set data path if data is not located in the current path.")
@click.option('-d', '--debug', default=False, is_flag=True, type=bool, help="Debug command-line parameters, do not run model.")
@click.option('-v', '--verbose', is_flag=True, default=False, type=bool, help="Enable verbose reporting.")
def recreat_util(data_path, verbose, debug):
    cli_model.model_set(ModelEnvironment.DataPath, data_path if data_path is not None else str(pathlib.Path().absolute()))    
    cli_model.model_set(ModelParameter.Verbosity, verbose)
    #new_model.is_debug = debug

@recreat_util.command(help="Specify land-use map")
@click.option('--nodata-values', default=[0.0], type=str, multiple=True, help="Nodata values in land-use raster.")
@click.argument('root-path')
@click.argument('landuse-filename')
def use(root_path, landuse_filename, nodata_values):
    # nodata values in the lsm are an argument to land use map alignment
    alignment_config = cli_model.get_task(CoreTask.Alignment)
    alignment_config.add_arg(ParameterNames.Alignment.NodataValues, sorted(list({float(num) for item in nodata_values for num in str(item).split(',')})))
    # root path and land use filename are required to setting land use in the model
    landuse_params = {
        ScenarioParameters.RootPath : root_path,
        ScenarioParameters.LanduseFileName : landuse_filename,
    }
    cli_model.model_set(ModelEnvironment.Scenario, landuse_params)

@recreat_util.command(help="Specify model parameters.")
@click.option('-p', '--patch', default=None, multiple=True, help="(Comma-separated) patch class(es).")
@click.option('-e', '--edge', default=None, multiple=True, help="(Comma-separated) edge class(es).")
@click.option('-g', '--buffered-edge', default=None, multiple=True, help="(Comma-separated) edge class(es) to buffer.")
@click.option('-b', '--built-up', default=None, multiple=True, help="(Comma-separated) built-up class(es).")
@click.option('-c', '--cost', default=None, multiple=True, help="(Comma-separated) cost(s).")
@click.option('--nodata-value', default=-9999, type=float, help="Set nodata value to be used in model exports.")
def params(cost, patch, edge, buffered_edge, built_up, nodata_value):
    # add classes to model
    cli_model.model_set(ClassType.Patch, sorted(list({int(num) for item in patch for num in str(item).split(',')})))
    cli_model.model_set(ClassType.Edge, sorted(list({int(num) for item in edge for num in str(item).split(',')})))
    cli_model.model_set(ClassType.BufferedEdge, sorted(list({int(num) for item in buffered_edge for num in str(item).split(',')})))
    cli_model.model_set(ClassType.Built_up, sorted(list({int(num) for item in built_up for num in str(item).split(',')})))
    # add costs to model
    cli_model.model_set(ModelParameter.Costs, sorted(list({int(num) for item in cost for num in str(item).split(',')})))
    # set nodata value to use
    cli_model.model_set(ModelParameter.NodataValue, nodata_value)


@recreat_util.command(help="Reclassify values in the land-use map prior to assessing landscape recreational potential")
@click.argument('source-classes')
@click.argument('destination-class')
def reclassify(source_classes, destination_class):       
    # this task is automatically created upon instantiation of the model class
    # we have to add reclassification mappings to this configuration
    current_config = cli_model.get_task(CoreTask.Alignment)   
    if current_config.get_arg(ParameterNames.Alignment.Mappings) is None:
        mappings = { int(destination_class) : sorted([int(item) for item in source_classes.split(',')]) }
        current_config.add_arg(ParameterNames.Alignment.Mappings, mappings)
    else:
        current_config.args[ParameterNames.Alignment.Mappings][int(destination_class)] = sorted([int(item) for item in source_classes.split(',')])        


@recreat_util.command(help="Identify clumps in land-use raster.")
@click.option('--barrier-classes', default=[0], type=str, multiple=True)
def clumps(barrier_classes):    
    new_task_config = Configuration(CoreTask.ClumpDetection)
    new_task_config.add_arg(ParameterNames.ClumpDetection.BarrierClasses, sorted(list({int(num) for item in barrier_classes for num in str(item).split(',')})))
    cli_model.add_task(new_task_config)

@recreat_util.command(help="Compute land-use (class) masks.")
def mask_landuses():
    new_task_config = Configuration(CoreTask.MaskLandUses)
    cli_model.add_task(new_task_config)

@recreat_util.command(help="Compute land-use (class) edges.")
@click.option('-i', '--ignore', type=float, default=None, help="Ignore edges to this class.")
def detect_edges(ignore):
    new_task_config = Configuration(CoreTask.EdgeDetection)
    new_task_config.add_arg(ParameterNames.EdgeDetection.LandUseClasses, cli_model.classes_edge)
    new_task_config.add_arg(ParameterNames.EdgeDetection.IgnoreEdgesToClass, ignore)
    new_task_config.add_arg(ParameterNames.EdgeDetection.BufferEdges, cli_model.classes_buffered_edges)    
    cli_model.add_task(new_task_config)

@recreat_util.command(help="Compute class total supply per cost.")
@click.option('-m', '--mode', type=click.Choice(['generic_filter', 'convolve', 'ocv_filter2d'], case_sensitive=True), default='ocv_filter2d')
def class_total_supply(mode):
    new_task_config = Configuration(CoreTask.ClassTotalSupply)
    new_task_config.add_arg(ParameterNames.ClassTotalSupply.Mode, mode)
    cli_model.add_task(new_task_config)

@recreat_util.command(help="Aggregate total supply per cost.")
@click.option('--landuse-weights', type=str, default=None)
@click.option('-u', '--exclude-non-weighted', is_flag=True, default=True, type=bool, help="Exclude export of non-weighted result(s).")
def aggregate_total_supply(landuse_weights, exclude_non_weighted):
    if landuse_weights is not None:
        landuse_weights = dict((int(x.strip()), float(y.strip()))
            for x, y in (element.split('=') 
            for element in landuse_weights.split(',')))
    
    new_task_config = Configuration(CoreTask.AggregateTotalSupply)
    new_task_config.add_arg(ParameterNames.AggregateClassTotalSupply.LandUseWeights, landuse_weights)
    new_task_config.add_arg(ParameterNames.AggregateClassTotalSupply.WriteNonWeightedResult, exclude_non_weighted)
    cli_model.add_task(new_task_config)
    
@recreat_util.command(help="Average total supply across costs.")
@click.option('-w', '--land-use-weighted', is_flag=True, default=False, type=bool, help="Use land-use-weighted supply as input.")
@click.option('--cost-weights', type=str, default=None)
@click.option('-s', '--exclude-scaled', is_flag=True, default=True, type=bool, help="Exclude export of scaled result(s).")
@click.option('-u', '--exclude-non-weighted', is_flag=True, default=True, type=bool, help="Exclude export of non-weighted result(s).")
def average_total_supply(cost_weights, land_use_weighted, exclude_non_weighted, exclude_scaled):
    if cost_weights is not None:
        cost_weights = dict((int(x.strip()), float(y.strip()))
             for x, y in (element.split('=') 
             for element in cost_weights.split(',')))
    
    new_task_config = Configuration(CoreTask.AverageTotalSupplyAcrossCost)
    new_task_config.add_arg(ParameterNames.AverageTotalSupplyAcrossCost.LandUseWeighting, land_use_weighted)
    new_task_config.add_arg(ParameterNames.AverageTotalSupplyAcrossCost.CostWeights, cost_weights)
    new_task_config.add_arg(ParameterNames.AverageTotalSupplyAcrossCost.WriteNonWeightedResult, exclude_non_weighted)
    new_task_config.add_arg(ParameterNames.AverageTotalSupplyAcrossCost.WriteScaledResult, exclude_scaled)
    cli_model.add_task(new_task_config)

@recreat_util.command(help="Compute class diversity per cost.")
def class_diversity():
    new_task_config = Configuration(CoreTask.ClassDiversity)
    cli_model.add_task(new_task_config)

@recreat_util.command(help="Average class diversity across costs.")
@click.option('--cost-weights', type=str, default=None)
@click.option('-s', '--exclude-scaled', is_flag=True, default=True, type=bool, help="Exclude export of scaled result(s).")
@click.option('-u', '--exclude-non-weighted', is_flag=True, default=True, type=bool, help="Exclude export of non-weighted result(s).")
def average_diversity(cost_weights, exclude_non_weighted, exclude_scaled):
    if cost_weights is not None:
        cost_weights = dict((int(x.strip()), float(y.strip()))
             for x, y in (element.split('=') 
             for element in cost_weights.split(',')))

    new_task_config = Configuration(CoreTask.AverageDiversityAcrossCost)
    new_task_config.add_arg(ParameterNames.AverageDiversityAcrossCost.CostWeights, cost_weights)
    new_task_config.add_arg(ParameterNames.AverageDiversityAcrossCost.WriteNonWeightedResult, exclude_non_weighted)
    new_task_config.add_arg(ParameterNames.AverageDiversityAcrossCost.WriteScaledResult, exclude_scaled)
    cli_model.add_task(new_task_config)

@recreat_util.command(help="Compute class-based flow per cost.")
def class_flow():
    new_task_config = Configuration(CoreTask.ClassFlow)
    cli_model.add_task(new_task_config)

@recreat_util.command(help="Compute proximity (distance) rasters.")
@click.option('-m', '--mode', type=click.Choice(['dr', 'xr'], case_sensitive=True), default='xr',  help="Method to use. Either distancerasters or xarray-spatial.")
@click.option('-b', '--include-builtup', is_flag=True, default=False, help="Include built-up in proximity assessment.")
def proximities(mode, include_builtup):
    new_task_config = Configuration(CoreTask.ComputeDistanceRasters)
    new_task_config.add_arg(ParameterNames.ComputeDistanceRasters.Mode, mode)
    new_task_config.add_arg(ParameterNames.ComputeDistanceRasters.LandUseClasses, None)
    new_task_config.add_arg(ParameterNames.ComputeDistanceRasters.AssessBuiltUp, include_builtup)
    cli_model.add_task(new_task_config)

@recreat_util.command(help="Disaggregate population.")
@click.option('-s', '--exclude-scaled', is_flag=True, default=True, type=bool, help="Exclude export of scaled result(s).")
@click.option('-m', '--method', default='saw', type=click.Choice(['saw', 'idm'], case_sensitive=True), help="Disaggregation method, either saw or idm, by default, saw.")
@click.option('-c', '--pixel-count', default=None, type=int, help="Number of built-up pixels per population raster cell.")
@click.option('-t', '--threshold', default=None, type=int, help="Mininmum class share per source unit (for idm only).")
@click.option('-n', '--sample-size', default=None, type=int, help="Mininmum sample size (for idm only).")
@click.argument('population-grid')
def disaggregate(population_grid, method, pixel_count, exclude_scaled, threshold, sample_size):
    new_task_config = Configuration(CoreTask.Disaggregation)
    new_task_config.add_arg(ParameterNames.Disaggregation.PopulationRaster, population_grid)
    new_task_config.add_arg(ParameterNames.Disaggregation.DisaggregationMethod, DisaggregationMethod(method))
    new_task_config.add_arg(ParameterNames.Disaggregation.ResidentialPixelToPopulationPixelCount, int(pixel_count))
    new_task_config.add_arg(ParameterNames.Disaggregation.WriteScaledResult, exclude_scaled)
    new_task_config.add_arg(ParameterNames.Disaggregation.ClassSampleThreshold, int(threshold))
    new_task_config.add_arg(ParameterNames.Disaggregation.MinimumSampleSize, sample_size)
    cli_model.add_task(new_task_config)
    
@recreat_util.command(help="Determine (average) cost to closest.")
@click.option('-d', '--max-distance', default=-1.0, type=float, help="Maximum cost value used for masking of cost rasters. If set to a negative value, do not mask areas with costs higher than maximum cost. Defaults to -1.")
@click.option('-b', '--mask-built-up', is_flag=True, default=False, type=bool, help="Indicates whether outputs will be restricted to built-up land-use classes, defaults to False.")
@click.option('-s', '--exclude-scaled', is_flag=True, default=True, type=bool, help="Exclude export of scaled result(s).")
def cost(max_distance, mask_built_up, exclude_scaled):
    new_task_config = Configuration(CoreTask.CostToClosest)
    new_task_config.add_arg(ParameterNames.CostToClosest.DistanceThreshold, max_distance)
    new_task_config.add_arg(ParameterNames.CostToClosest.MaskBuiltUp, mask_built_up)
    new_task_config.add_arg(ParameterNames.CostToClosest.WriteScaledResult, exclude_scaled)
    cli_model.add_task(new_task_config)




@recreat_util.result_callback()
def run_process(result, **kwargs):

    try:
        # model requirements check
        cli_model.validate()
    except ModelValidationError as e:
        print(f"recreat_util error: {Fore.RED}{Style.BRIGHT}{e}{Style.RESET_ALL}")
        return

    # print model summary
    cli_model.print() 
    
    # ask if model should be run
    user_confirm = input("Run this model? (y/N): ")
    user_confirm = False if user_confirm is None or user_confirm == '' or user_confirm.lower() == 'n' else True
    
    if not user_confirm:
        print('Aborted')
        return
    
    # run model
    cli_model.run()




if __name__ == '__main__':
    recreat_util()