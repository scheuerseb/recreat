###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################

from enum import Enum
from numpy import int32, float32, float64
from typing import List, Dict
from colorama import Fore, Back, Style
from rich import print as outp
from rich.panel import Panel
from rich.console import Console
from rich.table import Table

from .Configuration import Configuration

from .enumerations import *
from .exceptions import ModelValidationError

class recreat_model():

    params = None   
    tasks = None 

    debug = False

    def __init__(self) -> None:  
        
        self.params = {}         
        self.tasks = {}

    # Track model parameters, environment variables, etc.
    def model_set(self, parameter_type, value):
        if parameter_type not in self.params.keys():
            self.params[parameter_type] = None
        self.params[parameter_type] = value
        
    def model_get(self, parameter_type):
        if parameter_type not in self.params.keys():
            return None
        return self.params[parameter_type]
    

    def add_task(self, task_config: Configuration):    
        if task_config.task_type not in self.tasks.keys():
            self.tasks[task_config.task_type] = task_config

    def get_task(self, task_type) -> Configuration:
        return None if task_type not in self.tasks.keys() else self.tasks[task_type]

    
    def has_task_attached(self, task: any) -> bool:
        return task in self.tasks.keys()

    
    
    def getargs_model_params(self) -> List[Dict[str, any]]:
        return [
            {'param_name': ModelParameter.Verbosity.name(), 'param_value': self.verbose},
            {'param_name': ModelParameter.DataType.name(), 'param_value': self.datatype},
            {'param_name': ClassType.Patch.name(), 'param_value': self.classes_patch},
            {'param_name': ClassType.Edge.name(), 'param_value': self.classes_edge},
            {'param_name': ClassType.Built_up.name(), 'param_value': self.classes_builtup},
            {'param_name': ModelParameter.Costs.name(), 'param_value': self.costs},
        ]
    
   


    # Track classes defined in model
    # patches
    @property
    def classes_patch(self) -> List[int]:
        if ClassType.Patch not in self.params.keys():
            return []
        return self.params[ClassType.Patch]
   
    # edges
    @property
    def classes_edge(self) -> List[int]:
        all_edge_classes = []        
        if ClassType.Edge in self.params.keys():
            all_edge_classes += self.params[ClassType.Edge]
        if ClassType.BufferedEdge in self.params.keys():
            all_edge_classes += self.params[ClassType.BufferedEdge]        
        return all_edge_classes
    
    # buffered edges
    @property 
    def classes_buffered_edges(self) -> List[int]:
        if ClassType.BufferedEdge not in self.params.keys():
            return []
        return self.params[ClassType.BufferedEdge]
    
    # built-up classes
    @property 
    def classes_builtup(self) -> List[int]:
        if ClassType.Built_up not in self.params.keys():
            return []
        return self.params[ClassType.Built_up]
   

    # other model parameters
    # costs
    @property
    def costs(self) -> List[int]:
        if ModelParameter.Costs not in self.params.keys():
            return []
        return self.params[ModelParameter.Costs]    
    
    # datatype
    @property
    def datatype(self) -> str:
        return self.params[ModelParameter.DataType]
    # verbosity
    @property
    def verbose(self) -> bool:
        return self.params[ModelParameter.Verbosity]
    
    # model environment     
    # data-path 
    @property
    def data_path(self) -> str:
        return self.params[ModelEnvironment.DataPath]    
    

    # clean tmp folder
    @property
    def clean_temporary_files(self) -> bool:
        return self.environment[ModelEnvironment.clean_temporary_files]
    @clean_temporary_files.setter
    def clean_temporary_files(self, value: bool) -> None:
        self.model_set(self.environment, ModelEnvironment.clean_temporary_files, value)


    
   
    # is debugging
    @property 
    def is_debug(self):
        return self.debug

    
    
    
    @staticmethod
    def datatype_to_numpy(datatype_as_str: str) -> any:
        if datatype_as_str == 'int':
            return int32
        elif datatype_as_str == 'float':
            return float32
        elif datatype_as_str == 'double':
            return float64    
        else:
            return None    
    
    
        
    # aggregate supply
    def add_aggregate_supply(self, lu_weights:Dict[int,float], export_non_weighted: bool):        
        current_process = CoreTask.aggregate_class_total_supply
        self.add_task(current_process)
        self._add_process_config(current_process, recreat_process_parameters.lu_weights, lu_weights)
        self._add_process_config(current_process, recreat_process_parameters.export_non_weighted_results, export_non_weighted)

    # average total supply across cost
    def add_average_total_supply_across_cost(self, lu_weights:Dict[int,float], cost_weights: Dict[int,float], export_non_weighted: bool, export_scaled: bool) -> None:
        current_process = CoreTask.average_total_supply_across_cost
        self.add_task(current_process)
        self._add_process_config(current_process, recreat_process_parameters.lu_weights, lu_weights)
        self._add_process_config(current_process, recreat_process_parameters.cost_weights, cost_weights)
        self._add_process_config(current_process, recreat_process_parameters.export_non_weighted_results, export_non_weighted)
        self._add_process_config(current_process, recreat_process_parameters.export_scaled_results, export_scaled)


    # class diversity
    def add_class_diversity(self) -> None:
        self.add_task(CoreTask.class_diversity)

    # average diversity across cost
    def add_average_diversity_across_cost(self, cost_weights: Dict[int, float], export_non_weighted: bool, export_scaled: bool) -> None:
        current_process = CoreTask.average_diversity_across_cost
        self.add_task(current_process)
        self._add_process_config(current_process, recreat_process_parameters.cost_weights, cost_weights)
        self._add_process_config(current_process, recreat_process_parameters.export_non_weighted_results, export_non_weighted)
        self._add_process_config(current_process, recreat_process_parameters.export_scaled_results, export_scaled)

    # class flow
    def add_class_flow(self) -> None:
        self.add_task(CoreTask.class_flow)


    # proximity rasters
    def add_proximity(self, mode, lu_classes: List[int], include_builtup: bool) -> None:
        current_process = CoreTask.proximity
        self.add_task(current_process)
        self._add_process_config(current_process, recreat_process_parameters.classes_on_restriction, lu_classes)
        self._add_process_config(current_process, recreat_process_parameters.mode, mode)
        self._add_process_config(current_process, recreat_process_parameters.include_special_class, include_builtup)


    def add_average_cost(self, max_distance: float, mask_built_up: bool, export_scaled: bool) -> None:
        current_process = CoreTask.average_cost
        self.add_task(current_process)
        self._add_process_config(current_process, recreat_process_parameters.user_threshold, max_distance)
        self._add_process_config(current_process, recreat_process_parameters.include_special_class, mask_built_up)
        self._add_process_config(current_process, recreat_process_parameters.export_scaled_results, export_scaled)

    # population disaggregation
    def add_disaggregate_population(self, pop_raster: str,  force: bool, export_scaled: bool) -> None:
        current_process = CoreTask.population_disaggregation
        self.add_task(current_process)
        self._add_process_config(current_process, recreat_process_parameters.population_raster, pop_raster)
        self._add_process_config(current_process, recreat_process_parameters.export_scaled_results, export_scaled)
        self._add_process_config(current_process, recreat_process_parameters.force, force)

    def add_clustering_kmeans(self, dimensions, k, attempts):
        self.add_task(ClusteringTask.kmeans)
        # TODO Add more params




    def print(self) -> None:
        self._print_model_environment()
        self._print_land_use_map()
        self._print_model_classes()
        self._print_tasks()

    def _print_model_environment(self) -> None:
        outp(Panel("recreat model summary"))
        print(f"data path: {Fore.YELLOW}{Style.BRIGHT}{self.data_path}{Style.RESET_ALL}")
    
    def _print_land_use_map(self) -> None:
        if self.model_get(ModelEnvironment.LandUseMap) is not None:
            map_params = self.model_get(ModelEnvironment.LandUseMap)
            print(f"root path: {Fore.YELLOW}{Style.BRIGHT}{map_params[LandUseMapParameters.RootPath]}{Style.RESET_ALL}")
            print(f"use      : {Fore.CYAN}{Style.BRIGHT}{map_params[LandUseMapParameters.LanduseFileName]}{Style.RESET_ALL}")
            print(f"           {map_params[LandUseMapParameters.NodataValues]} -> {map_params[LandUseMapParameters.NodataFillValue]}")
    
    def _print_model_classes(self) -> None:
        # part 2: specified classes etc.
        print()
        tbl = Table(title="Parameter summary", show_lines=True)
        tbl.add_column("Parameter")
        tbl.add_column("Value(s)", style="cyan")
        tbl.add_row('Patch classes', ','.join(map(str, self.classes_patch)))
        tbl.add_row('Edge classes', ','.join(map(str, self.classes_edge)))
        tbl.add_row('Built-up classes', ','.join(map(str, self.classes_builtup)))
        tbl.add_row('Costs', ','.join(map(str, self.costs)))
        outp(tbl)

    def _print_tasks(self) -> None:
        print("Tasks:")            
        for p in CoreTask:
   
            if self.has_task_attached(p):
                print(f"{Fore.YELLOW}{Style.BRIGHT}   * {p.label()}{Style.RESET_ALL}")
                print(self.get_task(p).to_string())
            else:
                print(f"{Fore.WHITE}{Style.DIM}     {p.label()}{Style.RESET_ALL}")


    def validate(self):
        if self.costs is None or len(self.costs) < 1:
            raise ModelValidationError('No costs defined in model.')

    def run(self):
        
        from .assessment import Recreat
        from .clustering import kmeans

        rc = Recreat(self.data_path)
        for attrib in self.getargs_model_params():
            print(attrib)
            rc.set_params(**attrib)

        if set(list(CoreTask)).intersection(self.tasks.keys()):
            import_args = {k.name() : v for k,v in self.model_get(ModelEnvironment.LandUseMap).items()}
            print(import_args)
            rc.set_land_use_map(**import_args)

        # iterate over tasks
        p: CoreTask
        for p in CoreTask:

            if self.has_task_attached(p):
                task_args = {k.name() : v for k,v in self.get_task(p).args.items()}                
                # for debug purposes
                print(task_args)
                func = getattr(rc, p.name())
                func(**task_args)


                
        


