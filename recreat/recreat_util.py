###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################

from .recreat_environment import recreat_model, recreat_params, recreat_process, recreat_process_parameters

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
@click.option('-m', '--nodata', default=[0.0], type=str, multiple=True, help="Nodata values in land-use raster.")
@click.option('-f', '--fill', default=0, type=str, help="Fill value to replace nodata values in land-use raster.")
@click.option('--no-cleaning', is_flag=True, default=True, type=bool, help="Do not clean temporary files after completion.")
@click.argument('root-path')
@click.argument('landuse-filename')
def recreat_util(data_path, root_path, verbose, datatype, landuse_filename, nodata, fill, no_cleaning, debug):
    new_model.data_path = data_path if data_path is not None else str(pathlib.Path().absolute())
    new_model.root_path = root_path
    new_model.landuse_file = landuse_filename
    new_model.landuse_file_nodata_values = sorted(list({float(num) for item in nodata for num in str(item).split(',')}))
    new_model.nodata_fill_value = float(fill)
    new_model.verbose = verbose
    new_model.clean_temporary_files = no_cleaning
    new_model.datatype = datatype
    new_model.is_debug = debug

@recreat_util.command(help="Specify model parameters.")
@click.option('-p', '--patch', default=None, multiple=True, help="(Comma-separated) patch class(es).")
@click.option('-e', '--edge', default=None, multiple=True, help="(Comma-separated) edge class(es).")
@click.option('-g', '--buffer-edge', default=None, multiple=True, help="(Comma-separated) edge class(es) to buffer.")
@click.option('-b', '--built-up', default=None, multiple=True, help="(Comma-separated) built-up class(es).")
@click.option('-c', '--cost', default=None, multiple=True, help="(Comma-separated) cost(s).")
def params(cost, patch, edge, buffer_edge, built_up):
    new_model.classes_patch = sorted(list({int(num) for item in patch for num in str(item).split(',')}))
    new_model.classes_edge = sorted(list({int(num) for item in edge for num in str(item).split(',')}))
    new_model.classes_grow_edge = sorted(list({int(num) for item in buffer_edge for num in str(item).split(',')}))
    new_model.classes_builtup = sorted(list({int(num) for item in built_up for num in str(item).split(',')}))
    new_model.costs = sorted(list({int(num) for item in cost for num in str(item).split(',')}))


@recreat_util.command(help="Reclassify sets of classes into new class.")
#@click.option('-n', '--new', help="Value of new class.")
#@click.option('-o', '--old', multiple=True, help="(Comma-separated) class(es) to reclassify into new class.")
@click.argument('source-classes')
@click.argument('destination-class')
def reclassify(source_classes, destination_class):        
    new_model.add_reclassification(int(destination_class), sorted(list({int(num) for item in source_classes for num in str(item).split(',')})))
    
@recreat_util.command(help="Identify clumps in land-use raster.")
@click.option('--barrier-classes', default=[0], type=str, multiple=True)
def clumps(barrier_classes):    
    new_model.add_clump_detection(sorted(list({int(num) for item in barrier_classes for num in str(item).split(',')})))

@recreat_util.command(help="Compute land-use (class) masks.")
def mask_landuses():
    new_model.add_mask_landuses()

@recreat_util.command(help="Compute land-use (class) edges.")
@click.option('-i', '--ignore', type=float, default=None, help="Ignore edges to this class.")
def detect_edges(ignore):
    new_model.add_detect_edges(class_ignore_edges=ignore)

@recreat_util.command(help="Compute class total supply per cost.")
@click.option('-m', '--mode', type=click.Choice(['generic_filter', 'convolve', 'ocv_filter2d'], case_sensitive=False), default='ocv_filter2d')
def class_total_supply(mode):
    new_model.add_class_total_supply(mode)

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



@recreat_util.result_callback()
def run_process(result, **kwargs):
    
    print(new_model.classes_edge)
    
    user_confirm = new_model.get_model_confirmation()
    if not user_confirm:
        print('Aborted')
        return
    
    # conduct model initialization and process data as requested by user
    # instantiate
    
    from recreat import recreat    
    rc = recreat(new_model.data_path)

    # set parameters for model
    model_parameters = new_model.get_model_parameters()
    for p in recreat_params:
        rc.set_params(p.value, model_parameters[p])

    # import land-use map
    rc.set_land_use_map(new_model.root_path, new_model.landuse_file, new_model.landuse_file_nodata_values, new_model.nodata_fill_value)

    # conduct processing. This will be done in a sensible order depending on data requirements across tools        
    for p in recreat_process:
        if p in new_model.get_processes().keys():
            if p is recreat_process.reclassification:
                rc.reclassify(reclassifications=new_model.aggregations)
            
            if p is recreat_process.clump_detection:
                rc.detect_clumps(barrier_classes=new_model.get_processing_parameter(p, recreat_process_parameters.classes_on_restriction))
            
            if p is recreat_process.mask_landuses:
                rc.mask_landuses()
            
            if p is recreat_process.edge_detection:               
                rc.detect_edges(lu_classes=new_model.classes_edge,
                    ignore_edges_to_class=new_model.get_processing_parameter(p, recreat_process_parameters.classes_on_restriction),
                    buffer_edges=new_model.classes_grow_edge)
            
            if p is recreat_process.class_total_supply:
                rc.class_total_supply(mode = new_model.get_processing_parameter(p, recreat_process_parameters.mode))
            
            if p is recreat_process.aggregate_class_total_supply:
                rc.aggregate_class_total_supply(lu_weights=new_model.get_processing_parameter(p, recreat_process_parameters.lu_weights), 
                                                write_non_weighted_result=new_model.get_processing_parameter(p, recreat_process_parameters.export_non_weighted_results))
            
            if p is recreat_process.average_total_supply_across_cost:
                rc.average_total_supply_across_cost(lu_weights=new_model.get_processing_parameter(p, recreat_process_parameters.lu_weights), 
                                                    cost_weights=new_model.get_processing_parameter(p, recreat_process_parameters.cost_weights),
                                                    write_non_weighted_result=new_model.get_processing_parameter(p, recreat_process_parameters.export_non_weighted_results),
                                                    write_scaled_result=new_model.get_processing_parameter(p, recreat_process_parameters.export_scaled_results)) 
            
            if p is recreat_process.class_diversity:
                rc.class_diversity()
                
            if p is recreat_process.average_diversity_across_cost:
                rc.average_diversity_across_cost(cost_weights=new_model.get_processing_parameter(p, recreat_process_parameters.cost_weights),
                                                write_non_weighted_result=new_model.get_processing_parameter(p, recreat_process_parameters.export_non_weighted_results),
                                                write_scaled_result=new_model.get_processing_parameter(p, recreat_process_parameters.export_scaled_results))
                
            if p is recreat_process.proximity:
                rc.compute_distance_rasters(mode=new_model.get_processing_parameter(p, recreat_process_parameters.mode),
                                            lu_classes=new_model.get_processing_parameter(p, recreat_process_parameters.classes_on_restriction),
                                            assess_builtup=new_model.get_processing_parameter(p, recreat_process_parameters.include_special_class))
            
            if p is recreat_process.class_flow:
                rc.class_flow()

            if p is recreat_process.population_disaggregation:
                rc.disaggregate_population(population_grid=new_model.get_processing_parameter(p, recreat_process_parameters.population_raster),
                                           force_computing=new_model.get_processing_parameter(p, recreat_process_parameters.force),
                                           write_scaled_result=new_model.get_processing_parameter(p, recreat_process_parameters.export_scaled_results))



if __name__ == '__main__':
    recreat_util()