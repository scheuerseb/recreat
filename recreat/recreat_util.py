import click
import pathlib

recreat_environment = {}
recreat_environment['params'] = {}
recreat_environment['params']['use-data-type'] = None
recreat_environment['params']['verbose-reporting'] = False
recreat_environment['params']['classes.patch'] = []
recreat_environment['params']['classes.edge'] = []
recreat_environment['params']['classes.builtup'] = []
recreat_environment['params']['costs'] = []

recreat_environment['process'] = {}




#from recreat import recreat

@click.group(invoke_without_command=True, chain=True)
@click.option('-w', '--working-dir', default=None, help="Set working directory if data is not located in the current path.")
@click.option('-v', '--verbose', is_flag=True, default=False, type=bool, help="Enable verbose reporting.")
@click.option('--datatype', default=None, type=str, help="Set datatype to use. By default, the same datatype as the land-use raster is used.")
@click.argument('root_path')
@click.argument('land_use_filename')
def recreat_util(working_dir, root_path, verbose, datatype, land_use_filename):
    recreat_environment['working-directory'] = working_dir if working_dir is not None else str(pathlib.Path().absolute())
    recreat_environment['root-directory'] = root_path
    recreat_environment['land-use-file'] = land_use_filename
    recreat_environment['params']['verbose-reporting'] = verbose
    recreat_environment['params']['use-data-type'] = datatype

@recreat_util.command()
@click.option('-p', '--patch', default=None, multiple=True, help="Add pach class to model.")
@click.option('-e', '--edge', default=None, multiple=True, help="Add edge class to model.")
@click.option('-c', '--cost', default=None, multiple=True, help="Add cost to model.")
def params(cost, patch, edge):
    recreat_environment['params']['classes.patch'] = sorted(list({int(num) for item in patch for num in str(item).split(',')}))
    recreat_environment['params']['classes.edge'] = sorted(list({int(num) for item in edge for num in str(item).split(',')}))
    recreat_environment['params']['costs'] = sorted(list({int(num) for item in cost for num in str(item).split(',')}))

@recreat_util.result_callback()
def run_process(result, **kwargs):
    print('after')
    print(recreat_environment)
    print(kwargs)

@recreat_util.command()
@click.option('-n', '--new')
@click.option('-o', '--old', multiple=True)
def aggregate(new, old):       
    if not 'aggregate' in recreat_environment['process'].keys():
        recreat_environment['process']['aggregate'] = {}    
    recreat_environment['process']['aggregate'][int(new)] = sorted(list({int(num) for item in old for num in str(item).split(',')}))
    
@recreat_util.command()
def clumps():
    if not 'clumps' in recreat_environment['process'].keys():
        recreat_environment['process']['clumps'] = True

@recreat_util.command()
def class_supply():
    if not 'class-supply' in recreat_environment['process'].keys():
        recreat_environment['process']['class-supply'] = True

@recreat_util.command()
@click.option('--cw')
def average_class_supply(cw):
    print(cw)

if __name__ == '__main__':
    recreat_util()