###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################

from .Configuration import Configuration
from .enumerations import *
from .exceptions import ModelValidationError


from numpy import int32, float32, float64
from typing import List, Dict, Union
from colorama import Fore, Style

from rich import print as outp
from rich import box
from rich.panel import Panel
import os.path

class Model():

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
    


    def print(self) -> None:
        self._print_model_environment()
        self._print_land_use_map()
        self._print_model_classes()
        self._print_tasks()

    def _print_model_environment(self) -> None:
        outp(Panel("recreat model summary"))
        print(f"     data path: {Fore.YELLOW}{Style.BRIGHT}{self.data_path}{Style.RESET_ALL}")
    
    def _print_land_use_map(self) -> None:
        outp(Panel("Land-use data", box=box.SIMPLE))
        if self.model_get(ModelEnvironment.Scenario) is not None:
            map_params = self.model_get(ModelEnvironment.Scenario)
            print(f"     use {Fore.CYAN}{Style.BRIGHT}{map_params[ScenarioParameters.LanduseFileName]}{Style.RESET_ALL} in {map_params[ScenarioParameters.RootPath]}")
            print(f"         {map_params[ScenarioParameters.NodataValues]} -> {map_params[ScenarioParameters.NodataFillValue]}")
        else:
            print("Not defined in model.")

    def _print_model_classes(self) -> None:
        outp(Panel("Parameters", box=box.SIMPLE))
        print(f"     Patch classes    : {Fore.CYAN}{','.join(map(str, self.classes_patch))}{Style.RESET_ALL}")
        print(f"     Edge classes     : {Fore.CYAN}{','.join(map(str, self.classes_edge))}{Style.RESET_ALL}")
        print(f"     Built-up classes : {Fore.CYAN}{','.join(map(str, self.classes_builtup))}{Style.RESET_ALL}")
        print(f"     Costs            : {Fore.CYAN}{','.join(map(str, self.costs))}{Style.RESET_ALL}")
        

    def _print_tasks(self) -> None:
        outp(Panel("Tasks", box=box.SIMPLE))
       
        for p in CoreTask:   
            self._print_task_detail(p)
        for p in ClusteringTask:
            self._print_task_detail(p)
    
    def _print_task_detail(self, p: Union[CoreTask, ClusteringTask]) -> None:
        if self.has_task_attached(p):
            print(f"{Fore.YELLOW}{Style.BRIGHT}   * {p.label()}{Style.RESET_ALL}")
            print(self.get_task(p).to_string())
        else:
            print(f"{Fore.WHITE}{Style.DIM}     {p.label()}{Style.RESET_ALL}")

    def tasks_require_landuse_import(self):
        return len(set(list(CoreTask)).intersection(self.tasks.keys())) > 0

    def validate(self):
        if not os.path.isdir(self.data_path):
            raise ModelValidationError('Data-path does not exist in path.')
        #if len(self.classes_patch + self.classes_edge) < 1:
        #    raise ModelValidationError('No recreational classes defined in model.')
        #if self.costs is None or len(self.costs) < 1:
        #    raise ModelValidationError('No costs defined in model.')    
        if self.tasks_require_landuse_import():
            if self.model_get(ModelEnvironment.Scenario) is None:
                raise ModelValidationError('No land-use raster defined in model.')
            if not os.path.isdir(f"{self.data_path}/{self.model_get(ModelEnvironment.Scenario)[ScenarioParameters.RootPath]}"):
                raise ModelValidationError('Root-path does not exist in path.')            
            if not os.path.isfile(f"{self.data_path}/{self.model_get(ModelEnvironment.Scenario)[ScenarioParameters.RootPath]}/{self.model_get(ModelEnvironment.Scenario)[ScenarioParameters.LanduseFileName]}"):
                raise ModelValidationError('Specified land-use file does not exist in path.')            
        if self.has_task_attached(CoreTask.Disaggregation):
            if len(self.classes_builtup) < 1:
                raise ModelValidationError('No built-up classes defined in model for disaggregation task.')
            if not os.path.isfile(f"{self.data_path}/{self.model_get(ModelEnvironment.Scenario)[ScenarioParameters.RootPath]}/{self.get_task(CoreTask.Disaggregation).get_arg(ParameterNames.Disaggregation.PopulationRaster)}"):
                raise ModelValidationError('Specified population file does not exist in path.')
            
        

    def run_task(self, cli_model, p: Union[CoreTask, ClusteringTask]) -> None:
        # get task arguments
        task_args = {k.name() : v for k,v in self.get_task(p).args.items()}                
        # for debug purposes
        print(task_args)

        # execute corresponding method
        func = getattr(cli_model, p.name())
        func(**task_args)


    def run(self):
        
        from .assessment import Recreat
        from .clustering import kmeans

        rc = Recreat(self.data_path)
        for attrib in self.getargs_model_params():
            rc.set_params(**attrib)
                
        if self.tasks_require_landuse_import():
            import_args = {k.name() : v for k,v in self.model_get(ModelEnvironment.Scenario).items()}
            print(import_args)
            rc.set_land_use_map(**import_args)

        # iterate over tasks
        p: CoreTask
        for p in CoreTask:
            if self.has_task_attached(p):
                self.run_task(rc, p)

        p: ClusteringTask
        for p in ClusteringTask:
            if self.has_task_attached(p):
                self.run_task(rc, p)


                
        


