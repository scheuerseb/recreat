import click
import pathlib
from colorama import just_fix_windows_console
just_fix_windows_console()

from recreat_environment import recreat_model
from recreat import recreat

new_model = recreat_model()

#from recreat import recreat

@click.group(invoke_without_command=True, chain=True)
@click.option('-w', '--working-dir', default=None, help="Set working directory if data is not located in the current path.")
@click.option('-v', '--verbose', is_flag=True, default=False, type=bool, help="Enable verbose reporting.")
@click.option('--datatype', default=None, type=str, help="Set datatype to use. By default, the same datatype as the land-use raster is used.")
@click.option('-m', '--nodata', default=[0.0], type=str, multiple=True, help="Nodata values in land-use raster.")
@click.option('-f', '--fill', default=0, type=str, help="Fill value to replace nodata values in land-use raster.")
@click.argument('root-path')
@click.argument('landuse-filename')
def recreat_util(working_dir, root_path, verbose, datatype, landuse_filename, nodata, fill):
    new_model.data_path = working_dir if working_dir is not None else str(pathlib.Path().absolute())
    new_model.root_path = root_path
    new_model.landuse_file = landuse_filename
    new_model.verbose = verbose
    new_model.datatype = datatype

@recreat_util.command()
@click.option('-p', '--patch', default=None, multiple=True, help="(Comma-separated) patch class(es).")
@click.option('-e', '--edge', default=None, multiple=True, help="(Comma-separated) edge class(es).")
@click.option('-b', '--built-up', default=None, multiple=True, help="(Comma-separated) built-up class(es).")
@click.option('-c', '--cost', default=None, multiple=True, help="(Comma-separated) cost(s).")
def params(cost, patch, edge, built_up):
    new_model.classes_patch = sorted(list({int(num) for item in patch for num in str(item).split(',')}))
    new_model.classes_edge = sorted(list({int(num) for item in edge for num in str(item).split(',')}))
    new_model.classes_builtup = sorted(list({int(num) for item in built_up for num in str(item).split(',')}))
    new_model.costs = sorted(list({int(num) for item in cost for num in str(item).split(',')}))


@recreat_util.command()
@click.option('-n', '--new', help="Value of new class.")
@click.option('-o', '--old', multiple=True, help="(Comma-separated) class(es) to aggregate into new class.")
def aggregate(new, old):        
    new_model.add_aggregation(int(new), sorted(list({int(num) for item in old for num in str(item).split(',')})))
    
@recreat_util.command()
@click.option('--barrier-classes', default=[0], type=str, multiple=True)
def clump(barrier_classes):    
    new_model.add_clump_detection(sorted(list({int(num) for item in barrier_classes for num in str(item).split(',')})))

@recreat_util.command()
def mask():
    new_model.add_mask_landuses()


@recreat_util.command()
@click.option('--mode', type=str, default='ocv_filter2d')
def class_total_supply(mode):
    new_model.add_class_total_supply(mode)


@recreat_util.command()
@click.option('--landuse-weights', type=str, default=None)
@click.option('--cost-weights', type=str, default=None)
@click.option('-s', '--exclude-scaled', is_flag=True, default=True, type=bool, help="Exclude export of scaled result(s).")
@click.option('-u', '--exclude-non-weighted', is_flag=True, default=True, type=bool, help="Exclude export of non-weighted result(s).")
def average_supply(cost_weights, landuse_weights, exclude_non_weighted, exclude_scaled):
    if cost_weights is not None:
        cost_weights = dict((int(x.strip()), float(y.strip()))
             for x, y in (element.split('=') 
             for element in cost_weights.split(',')))
    if landuse_weights is not None:
        landuse_weights = dict((int(x.strip()), float(y.strip()))
            for x, y in (element.split('=') 
            for element in landuse_weights.split(',')))
    
    new_model.add_average_total_supply_across_cost(cost_weights=cost_weights, lu_weights=landuse_weights, export_non_weighted=exclude_non_weighted, export_scaled=exclude_scaled)


@recreat_util.command()
def class_diversity():
    new_model.add_class_diversity()

@recreat_util.command()
@click.option('--cost-weights', type=str, default=None)
@click.option('-s', '--exclude-scaled', is_flag=True, default=True, type=bool, help="Exclude export of scaled result(s).")
@click.option('-u', '--exclude-non-weighted', is_flag=True, default=True, type=bool, help="Exclude export of non-weighted result(s).")
def average_diversity(cost_weights, exclude_non_weighted, exclude_scaled):
    if cost_weights is not None:
        cost_weights = dict((int(x.strip()), float(y.strip()))
             for x, y in (element.split('=') 
             for element in cost_weights.split(',')))
    new_model.add_average_diversity_across_cost(cost_weights=cost_weights, export_non_weighted=exclude_non_weighted, export_scaled=exclude_scaled)


@recreat_util.command()
def class_flow():
    new_model.add_class_flow()

@recreat_util.result_callback()
def run_process(result, **kwargs):
    user_confirm = new_model.get_model_confirmation()
    if not user_confirm:
        print('Aborted')
        return
    
    # conduct model initialization
    rc = recreat(new_model.data_path)

    # conduct model processing
    processes = new_model.get_processes()
    for p in processes:
        print(p)




if __name__ == '__main__':
    recreat_util()