###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################

from .model import recreat_model, ModelParameter, ModelEnvironment, LandUseMapParameters, CoreTask, ClusteringTask
from .model import ClassType
from .Configuration import Configuration
from .parameternames import ParameterNames


import click
import pathlib
from colorama import just_fix_windows_console
just_fix_windows_console()




new_model = recreat_model()

@click.group(invoke_without_command=True, chain=True, )
@click.option('-w', '--data-path', default=None, help="Set data path if data is not located in the current path.")
@click.option('-d', '--debug', default=False, is_flag=True, type=bool, help="Debug command-line parameters, do not run model.")
@click.option('-v', '--verbose', is_flag=True, default=False, type=bool, help="Enable verbose reporting.")
@click.option('--datatype', default=None, type=click.Choice(['int', 'float', 'double'], case_sensitive=False), help="Set datatype to use. By default, the same datatype as the land-use raster is used.")
@click.option('--no-cleaning', is_flag=True, default=True, type=bool, help="Do not clean temporary files after completion.")
def recreat_util(data_path, verbose, datatype, no_cleaning, debug):
    new_model.model_set(ModelParameter.Verbosity, verbose)
    new_model.model_set(ModelParameter.DataType, recreat_model.datatype_to_numpy(datatype) )
    new_model.model_set(ModelEnvironment.DataPath, data_path if data_path is not None else str(pathlib.Path().absolute()))
    #new_model.clean_temporary_files = no_cleaning
    #new_model.is_debug = debug

@recreat_util.command(help="Use specified land-use dataset")
@click.option('-m', '--nodata', default=[0.0], type=str, multiple=True, help="Nodata values in land-use raster.")
@click.option('-f', '--fill', default=0, type=str, help="Fill value to replace nodata values in land-use raster.")
@click.argument('root-path')
@click.argument('landuse-filename')
def use(root_path, landuse_filename, nodata, fill):
    landuse_params = {
        LandUseMapParameters.RootPath.value : root_path,
        LandUseMapParameters.LanduseFileName.value : landuse_filename,
        LandUseMapParameters.NodataValues.value : sorted(list({float(num) for item in nodata for num in str(item).split(',')})),
        LandUseMapParameters.NodataFillValue.value : float(fill)
    }
    new_model.model_set(ModelEnvironment.LandUseMap, landuse_params)


@recreat_util.command(help="Specify model parameters.")
@click.option('-p', '--patch', default=None, multiple=True, help="(Comma-separated) patch class(es).")
@click.option('-e', '--edge', default=None, multiple=True, help="(Comma-separated) edge class(es).")
@click.option('-g', '--buffered-edge', default=None, multiple=True, help="(Comma-separated) edge class(es) to buffer.")
@click.option('-b', '--built-up', default=None, multiple=True, help="(Comma-separated) built-up class(es).")
@click.option('-c', '--cost', default=None, multiple=True, help="(Comma-separated) cost(s).")
def params(cost, patch, edge, buffered_edge, built_up):
    # add classes to model
    new_model.model_set(ClassType.Patch, sorted(list({int(num) for item in patch for num in str(item).split(',')})))
    new_model.model_set(ClassType.Edge, sorted(list({int(num) for item in edge for num in str(item).split(',')})))
    new_model.model_set(ClassType.BufferedEdge, sorted(list({int(num) for item in buffered_edge for num in str(item).split(',')})))
    new_model.model_set(ClassType.Built_up, sorted(list({int(num) for item in built_up for num in str(item).split(',')})))
    # add costs to model
    new_model.model_set(ModelParameter.Costs, sorted(list({int(num) for item in cost for num in str(item).split(',')})))


@recreat_util.command(help="Reclassify sets of classes into new class.")
@click.option('-e', '--export', default=None, type=str, help="Export result of reclassification into root-path.")
@click.argument('source-classes')
@click.argument('destination-class')
def reclassify(source_classes, destination_class, export):       
    current_config = new_model.get_task(CoreTask.Reclassification)   
    if current_config is None:
        mappings = { int(destination_class) : sorted([int(item) for item in source_classes.split(',')]) }
        new_task_config = Configuration(CoreTask.Reclassification)
        new_task_config.add_arg(ParameterNames.Reclassification.Mappings.value, mappings)
        new_task_config.add_arg(ParameterNames.Reclassification.ExportFilename.value, export)        
        new_model.add_task(new_task_config)
    else:
        current_config.args[ParameterNames.Reclassification.Mappings.value][int(destination_class)] = sorted([int(item) for item in source_classes.split(',')])
        if (
            (current_config.args[ParameterNames.Reclassification.ExportFilename.value] is not None
            and export is not None)
            or current_config.args[ParameterNames.Reclassification.ExportFilename.value] is None
        ):
            current_config.args[ParameterNames.Reclassification.ExportFilename.value] = export

        
@recreat_util.command(help="Identify clumps in land-use raster.")
@click.option('--barrier-classes', default=[0], type=str, multiple=True)
def clumps(barrier_classes):    
    new_task_config = Configuration(CoreTask.ClumpDetection)
    new_task_config.add_arg(ParameterNames.ClumpDetection.BarrierClasses.value, sorted(list({int(num) for item in barrier_classes for num in str(item).split(',')})))
    new_model.add_task(new_task_config)


@recreat_util.command(help="Compute land-use (class) masks.")
def mask_landuses():
    new_task_config = Configuration(CoreTask.MaskLandUses)
    new_model.add_task(new_task_config)

@recreat_util.command(help="Compute land-use (class) edges.")
@click.option('-i', '--ignore', type=float, default=None, help="Ignore edges to this class.")
def detect_edges(ignore):
    new_task_config = Configuration(CoreTask.EdgeDetection)
    new_task_config.add_arg(ParameterNames.EdgeDetection.LandUseClasses.value, new_model.classes_edge)
    new_task_config.add_arg(ParameterNames.EdgeDetection.IgnoreEdgesToClass.value, ignore)
    new_task_config.add_arg(ParameterNames.EdgeDetection.BufferEdges.value, new_model.classes_buffered_edges)    
    new_model.add_task(new_task_config)

@recreat_util.command(help="Compute class total supply per cost.")
@click.option('-m', '--mode', type=click.Choice(['generic_filter', 'convolve', 'ocv_filter2d'], case_sensitive=True), default='ocv_filter2d')
def class_total_supply(mode):
    new_task_config = Configuration(CoreTask.ClassTotalSupply)
    new_task_config.add_arg(ParameterNames.ClassTotalSupply.Mode.value, mode)
    new_model.add_task(new_task_config)

@recreat_util.command(help="Aggregate total supply per cost.")
@click.option('--landuse-weights', type=str, default=None)
@click.option('-u', '--exclude-non-weighted', is_flag=True, default=True, type=bool, help="Exclude export of non-weighted result(s).")
def aggregate_total_supply(landuse_weights, exclude_non_weighted):
    if landuse_weights is not None:
        landuse_weights = dict((int(x.strip()), float(y.strip()))
            for x, y in (element.split('=') 
            for element in landuse_weights.split(',')))
    new_model.add_aggregate_supply(lu_weights=landuse_weights, export_non_weighted=exclude_non_weighted)

@recreat_util.command(help="Average total supply across costs.")
@click.option('--landuse-weights', type=str, default=None)
@click.option('--cost-weights', type=str, default=None)
@click.option('-s', '--exclude-scaled', is_flag=True, default=True, type=bool, help="Exclude export of scaled result(s).")
@click.option('-u', '--exclude-non-weighted', is_flag=True, default=True, type=bool, help="Exclude export of non-weighted result(s).")
def average_total_supply(cost_weights, landuse_weights, exclude_non_weighted, exclude_scaled):
    if cost_weights is not None:
        cost_weights = dict((int(x.strip()), float(y.strip()))
             for x, y in (element.split('=') 
             for element in cost_weights.split(',')))
    if landuse_weights is not None:
        landuse_weights = dict((int(x.strip()), float(y.strip()))
            for x, y in (element.split('=') 
            for element in landuse_weights.split(',')))
    
    new_model.add_average_total_supply_across_cost(cost_weights=cost_weights, lu_weights=landuse_weights, export_non_weighted=exclude_non_weighted, export_scaled=exclude_scaled)


@recreat_util.command(help="Compute class diversity per cost.")
def class_diversity():
    new_model.add_class_diversity()

@recreat_util.command(help="Average class diversity across costs.")
@click.option('--cost-weights', type=str, default=None)
@click.option('-s', '--exclude-scaled', is_flag=True, default=True, type=bool, help="Exclude export of scaled result(s).")
@click.option('-u', '--exclude-non-weighted', is_flag=True, default=True, type=bool, help="Exclude export of non-weighted result(s).")
def average_diversity(cost_weights, exclude_non_weighted, exclude_scaled):
    if cost_weights is not None:
        cost_weights = dict((int(x.strip()), float(y.strip()))
             for x, y in (element.split('=') 
             for element in cost_weights.split(',')))

    new_model.add_average_diversity_across_cost(cost_weights=cost_weights, export_non_weighted=exclude_non_weighted, export_scaled=exclude_scaled)


@recreat_util.command(help="Compute class-based flow per cost.")
def class_flow():
    new_model.add_class_flow()

@recreat_util.command(help="Compute proximity (distance) rasters.")
@click.option('-m', '--mode', type=click.Choice(['dr', 'xr']), default='xr', help="Method to use. Either distancerasters or xarray-spatial.")
@click.option('-b', '--include-builtup', is_flag=True, default=False, help="Include built-up in proximity assessment.")
def proximities(mode, include_builtup):
    new_model.add_proximity(mode=mode, lu_classes=None, include_builtup=include_builtup)

@recreat_util.command(help="Disaggregate population to built-up.")
@click.option('-s', '--exclude-scaled', is_flag=True, default=True, type=bool, help="Exclude export of scaled result(s).")
@click.option('-f', '--force', is_flag=True, default=False, type=bool, help="Force recomputation of intermediate products.")
@click.argument('pop')
def disaggregate_population(pop, exclude_scaled, force):
    new_model.add_disaggregate_population(pop_raster=pop, force=force, export_scaled=exclude_scaled)

@recreat_util.command(help="Determine (average) cost to closest.")
@click.option('-d', '--max-distance', default=-1.0, type=float, help="Maximum cost value used for masking of cost rasters. If set to a negative value, do not mask areas with costs higher than maximum cost. Defaults to -1.")
@click.option('-b', '--mask-built-up', is_flag=True, default=False, type=bool, help="Indicates whether outputs will be restricted to built-up land-use classes, defaults to False.")
@click.option('-s', '--exclude-scaled', is_flag=True, default=True, type=bool, help="Exclude export of scaled result(s).")
def cost(max_distance, mask_built_up, exclude_scaled):
    new_model.add_average_cost(max_distance=max_distance, mask_built_up=mask_built_up, export_scaled=exclude_scaled)

@recreat_util.result_callback()
def run_process(result, **kwargs):

    # print model summary
    new_model.print() 
    # ask if model should be run
    user_confirm = input("Run this model? (y/N): ")
    user_confirm = False if user_confirm is None or user_confirm == '' or user_confirm.lower() == 'n' else True
    if not user_confirm:
        print('Aborted')
        return
    
    # run model
    new_model.run()



            
    #         if p is CoreTask.edge_detection:               
    #             rc.detect_edges(lu_classes=new_model.classes_edge,
    #                 ignore_edges_to_class=new_model.get_processing_parameter(p, recreat_process_parameters.classes_on_restriction),
    #                 buffer_edges=new_model.classes_buffered_edges)
            
    #         if p is CoreTask.class_total_supply:
    #             rc.class_total_supply(mode = new_model.get_processing_parameter(p, recreat_process_parameters.mode))
            
    #         if p is CoreTask.aggregate_class_total_supply:
    #             rc.aggregate_class_total_supply(lu_weights=new_model.get_processing_parameter(p, recreat_process_parameters.lu_weights), 
    #                                             write_non_weighted_result=new_model.get_processing_parameter(p, recreat_process_parameters.export_non_weighted_results))
            
    #         if p is CoreTask.average_total_supply_across_cost:
    #             rc.average_total_supply_across_cost(lu_weights=new_model.get_processing_parameter(p, recreat_process_parameters.lu_weights), 
    #                                                 cost_weights=new_model.get_processing_parameter(p, recreat_process_parameters.cost_weights),
    #                                                 write_non_weighted_result=new_model.get_processing_parameter(p, recreat_process_parameters.export_non_weighted_results),
    #                                                 write_scaled_result=new_model.get_processing_parameter(p, recreat_process_parameters.export_scaled_results)) 
            
    #         if p is CoreTask.class_diversity:
    #             rc.class_diversity()
                
    #         if p is CoreTask.average_diversity_across_cost:
    #             rc.average_diversity_across_cost(cost_weights=new_model.get_processing_parameter(p, recreat_process_parameters.cost_weights),
    #                                             write_non_weighted_result=new_model.get_processing_parameter(p, recreat_process_parameters.export_non_weighted_results),
    #                                             write_scaled_result=new_model.get_processing_parameter(p, recreat_process_parameters.export_scaled_results))
                
    #         if p is CoreTask.proximity:
    #             rc.compute_distance_rasters(mode=new_model.get_processing_parameter(p, recreat_process_parameters.mode),
    #                                         lu_classes=new_model.get_processing_parameter(p, recreat_process_parameters.classes_on_restriction),
    #                                         assess_builtup=new_model.get_processing_parameter(p, recreat_process_parameters.include_special_class))
            
    #         if p is CoreTask.class_flow:
    #             rc.class_flow()

    #         if p is CoreTask.population_disaggregation:
    #             rc.disaggregate_population(population_grid=new_model.get_processing_parameter(p, recreat_process_parameters.population_raster),
    #                                        force_computing=new_model.get_processing_parameter(p, recreat_process_parameters.force),
    #                                        write_scaled_result=new_model.get_processing_parameter(p, recreat_process_parameters.export_scaled_results))



if __name__ == '__main__':
    recreat_util()