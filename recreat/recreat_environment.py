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

class recreat_process(Enum):
    reclassification = 'reclassification'
    clump_detection = 'clumps'
    mask_landuses = 'mask-landuses'
    edge_detection = 'detect-edges'
    class_total_supply = 'class-total-supply'
    aggregate_class_total_supply = 'aggregate-total-supply'
    average_total_supply_across_cost = 'average-total-supply'
    class_diversity = 'class-diversity'
    average_diversity_across_cost = 'average-diversity'
    population_disaggregation = 'disaggregate-population'
    class_flow = 'class-flow'
    proximity = 'proximities'
    average_cost = 'average-cost'

class clustering_process(Enum):
    kmeans = 'kmeans'

class recreat_params(Enum):
    data_type = 'use-data-type'
    verbose = 'verbose-reporting'
    classes_patch = 'classes.patch'
    classes_edge = 'classes.edge'    
    classes_builtup = 'classes.builtup'
    costs = 'costs'

class recreat_env(Enum):
    data_path = 'working-directory'
    root = 'root-directory'
    landuse_file = 'land-use-file'
    nodata_values = 'land-use-file-nodata-values'
    nodata_fill_value = 'nodata-fill-value'
    clean_temporary_files = 'clean-temp-path'

class recreat_process_parameters(Enum):
    classes_on_restriction = 'classes-on-restriction'
    buffered_edge_classes = 'buffered-edge-classes'
    lu_weights = 'landuse-weights'
    cost_weights = 'cost-weights'
    mode = 'mode'
    export_name = 'export-name'
    export_non_weighted_results = 'export-non-weighted-results'
    export_scaled_results = 'export-scaled-results'
    include_special_class = 'include-special-class'
    population_raster = 'population-grid'
    force = 'force'
    user_threshold = 'user-threshold'
    

class recreat_model():

    params = None   
    processes = None 
    debug = False
    environment = None

    def __init__(self) -> None:  
        
        # initialize with sensible default values      
        
        self.params = {} 
        self.environment = {}
        self.processes = {}

        self.params[recreat_params.data_type] = None
        self.params[recreat_params.verbose] = False
        self.params[recreat_params.classes_patch] = []
        self.params[recreat_params.classes_edge] = []
        self.params[recreat_params.classes_builtup] = []
        self.params[recreat_params.costs] = []

        self.specials = {}
        self.specials['grow-edges'] = None

    # data path and root path
    @property
    def data_path(self) -> str:
        return self.environment[recreat_env.data_path]
    @data_path.setter
    def data_path(self, data_path: str) -> None:
        self._add_key(self.environment, recreat_env.data_path, data_path)
    @property
    def root_path(self) -> str:
        return self.environment[recreat_env.root]
    @root_path.setter
    def root_path(self, root_dir: str) -> None:
        self._add_key(self.environment, recreat_env.root, root_dir)

    # clean tmp folder
    @property
    def clean_temporary_files(self) -> bool:
        return self.environment[recreat_env.clean_temporary_files]
    @clean_temporary_files.setter
    def clean_temporary_files(self, value: bool) -> None:
        self._add_key(self.environment, recreat_env.clean_temporary_files, value)

    # landuse file
    @property
    def landuse_file(self) -> str:
        return self.environment[recreat_env.landuse_file]
    @landuse_file.setter
    def landuse_file(self, filename: str) -> None:
        self._add_key(self.environment, recreat_env.landuse_file, filename)
    @property
    def landuse_file_nodata_values(self) -> List[float]:
        return self.environment[recreat_env.nodata_values]
    @landuse_file_nodata_values.setter
    def landuse_file_nodata_values(self, nodata_vals: List[float]) -> None:
        self._add_key(self.environment, recreat_env.nodata_values, nodata_vals)
    @property
    def nodata_fill_value(self) -> float:
        return self.environment[recreat_env.nodata_fill_value]
    @nodata_fill_value.setter
    def nodata_fill_value(self, value: float) -> None:
        self._add_key(self.environment, recreat_env.nodata_fill_value, value)

    # debug model
    @property 
    def is_debug(self):
        return self.debug
    @is_debug.setter
    def is_debug(self, value):
        self.debug = value

    # verbosity
    @property
    def verbose(self) -> bool:
        return self.params[recreat_params.verbose]
    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._add_key(self.params, recreat_params.verbose, value)
    
    # datatype
    @property
    def datatype(self) -> str:
        return self.params[recreat_params.data_type]
    @datatype.setter
    def datatype(self, value: str) -> None:
        self._add_key(self.params, recreat_params.data_type, value)
    def datatype_as_numpy(self) -> any:
        if self.datatype == 'int':
            return int32
        elif self.datatype == 'float':
            return float32
        elif self.datatype == 'double':
            return float64        
    
    # patch classes
    @property
    def classes_patch(self) -> List[int]:
        return self.params[recreat_params.classes_patch]
    @classes_patch.setter
    def classes_patch(self, class_values: List[int]):
        self._add_key(self.params, recreat_params.classes_patch, class_values)
    
    # edge classes
    @property
    def classes_edge(self) -> List[int]:
        retval = []        
        retval += self.params[recreat_params.classes_edge]
        if self.specials['grow-edges'] is not None:
            retval += self.specials['grow-edges']
        return retval    
    @classes_edge.setter
    def classes_edge(self, class_values: List[int]):
        self._add_key(self.params, recreat_params.classes_edge, class_values)
    
    # grow edges
    @property 
    def buffered_edge_classes(self) -> List[int]:
        return self.specials['grow-edges']     
    @buffered_edge_classes.setter
    def buffered_edge_classes(self, class_values: List[int]) -> None:
        if len(class_values) == 0:
            class_values = None
        self.specials['grow-edges'] = class_values 
    
    # builtup classes
    @property 
    def classes_builtup(self) -> List[int]:
        return self.params[recreat_params.classes_builtup]
    @classes_builtup.setter
    def classes_builtup(self, class_values: List[int]):
        self._add_key(self.params, recreat_params.classes_builtup, class_values)

    # costs
    @property
    def costs(self) -> List[int]:
        return self.params[recreat_params.costs]
    @costs.setter
    def costs(self, cost_values: List[int]):
        self._add_key(self.params, recreat_params.costs, cost_values)
        


    # processes and their parameters
    
    # reclassification
    def add_reclassification(self, dest_class: int, source_classes: List[int], export_filename) -> None:
        current_process = recreat_process.reclassification
        self._add_process(current_process)

        # assert if previously, an export has been requested. In that case, do not overwrite with non value
        # in case that two conflicting names were defined, simply use the latter name for now
        if self.get_processing_parameter(current_process, recreat_process_parameters.export_name) is None:
            # overwrite with whatever, even with None
            self._add_process_config(current_process, recreat_process_parameters.export_name, export_filename)     
        else:
            # overwrite only if not None
            if export_filename is not None:
                self.processes[current_process][recreat_process_parameters.export_name] = export_filename     

        self.processes[current_process]['classes'][dest_class] = source_classes
    
        

    @property
    def mappings(self) -> Dict[int, List[int]]:
        return self.processes[recreat_process.reclassification]['classes']
    
    # clump detection
    def add_clump_detection(self, barrier_classes: List[int]) -> None:
        current_process = recreat_process.clump_detection
        self._add_process(current_process)
        self._add_process_config(current_process, recreat_process_parameters.classes_on_restriction, barrier_classes)

    # landuse masking
    def add_mask_landuses(self) -> None:        
        self._add_process(recreat_process.mask_landuses)

    # edge detection
    def add_detect_edges(self, class_ignore_edges: float = 0) -> None:
        current_process = recreat_process.edge_detection
        self._add_process(current_process)
        self._add_process_config(current_process, recreat_process_parameters.classes_on_restriction, class_ignore_edges)

    # class total supply
    def add_class_total_supply(self, mode: str) -> None:
        current_process = recreat_process.class_total_supply
        self._add_process(current_process)
        self._add_process_config(current_process, recreat_process_parameters.mode, mode)

    # aggregate supply
    def add_aggregate_supply(self, lu_weights:Dict[int,float], export_non_weighted: bool):        
        current_process = recreat_process.aggregate_class_total_supply
        self._add_process(current_process)
        self._add_process_config(current_process, recreat_process_parameters.lu_weights, lu_weights)
        self._add_process_config(current_process, recreat_process_parameters.export_non_weighted_results, export_non_weighted)

    # average total supply across cost
    def add_average_total_supply_across_cost(self, lu_weights:Dict[int,float], cost_weights: Dict[int,float], export_non_weighted: bool, export_scaled: bool) -> None:
        current_process = recreat_process.average_total_supply_across_cost
        self._add_process(current_process)
        self._add_process_config(current_process, recreat_process_parameters.lu_weights, lu_weights)
        self._add_process_config(current_process, recreat_process_parameters.cost_weights, cost_weights)
        self._add_process_config(current_process, recreat_process_parameters.export_non_weighted_results, export_non_weighted)
        self._add_process_config(current_process, recreat_process_parameters.export_scaled_results, export_scaled)


    # class diversity
    def add_class_diversity(self) -> None:
        self._add_process(recreat_process.class_diversity)

    # average diversity across cost
    def add_average_diversity_across_cost(self, cost_weights: Dict[int, float], export_non_weighted: bool, export_scaled: bool) -> None:
        current_process = recreat_process.average_diversity_across_cost
        self._add_process(current_process)
        self._add_process_config(current_process, recreat_process_parameters.cost_weights, cost_weights)
        self._add_process_config(current_process, recreat_process_parameters.export_non_weighted_results, export_non_weighted)
        self._add_process_config(current_process, recreat_process_parameters.export_scaled_results, export_scaled)

    # class flow
    def add_class_flow(self) -> None:
        self._add_process(recreat_process.class_flow)


    # proximity rasters
    def add_proximity(self, mode, lu_classes: List[int], include_builtup: bool) -> None:
        current_process = recreat_process.proximity
        self._add_process(current_process)
        self._add_process_config(current_process, recreat_process_parameters.classes_on_restriction, lu_classes)
        self._add_process_config(current_process, recreat_process_parameters.mode, mode)
        self._add_process_config(current_process, recreat_process_parameters.include_special_class, include_builtup)


    def add_average_cost(self, max_distance: float, mask_built_up: bool, export_scaled: bool) -> None:
        current_process = recreat_process.average_cost
        self._add_process(current_process)
        self._add_process_config(current_process, recreat_process_parameters.user_threshold, max_distance)
        self._add_process_config(current_process, recreat_process_parameters.include_special_class, mask_built_up)
        self._add_process_config(current_process, recreat_process_parameters.export_scaled_results, export_scaled)

    # population disaggregation
    def add_disaggregate_population(self, pop_raster: str,  force: bool, export_scaled: bool) -> None:
        current_process = recreat_process.population_disaggregation
        self._add_process(current_process)
        self._add_process_config(current_process, recreat_process_parameters.population_raster, pop_raster)
        self._add_process_config(current_process, recreat_process_parameters.export_scaled_results, export_scaled)
        self._add_process_config(current_process, recreat_process_parameters.force, force)

    def add_clustering_kmeans(self, dimensions, k, attempts):
        self._add_process(clustering_process.kmeans)
        # TODO Add more params

    def get_model_confirmation(self) -> bool:
        # print a summary
        # part 1: general model information
        outp(Panel("recreat model summary"))
        print(f"data path: {Fore.YELLOW}{Style.BRIGHT}{self.data_path}{Style.RESET_ALL}")
        print(f"root path: {Fore.YELLOW}{Style.BRIGHT}{self.root_path}{Style.RESET_ALL}")

        if self.landuse_file:
            print(f"use      : {Fore.CYAN}{Style.BRIGHT}{self.landuse_file}{Style.RESET_ALL}")
            print(f"           {self.landuse_file_nodata_values} -> {self.nodata_fill_value}")

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

        # part 3: processes and process details
        print()
        print("Process:")

        for p in recreat_process:
            processes_info = self._get_process_info(p)
            for outstr in processes_info:
                print(outstr)
        
        print()
        if self.is_debug:
            return False
        else:
            user_confirm = input("Run this model? (y/N): ")
            if user_confirm is None or user_confirm == '' or user_confirm.lower() == 'n':
                return False
            else:
                return True
          
    def _get_process_info(self, p: recreat_process) -> List[str]:
        
        str_results = []
        contains_process = p in self.processes.keys()

        star = '*' if contains_process else ' '
        process_foreground = Fore.YELLOW if contains_process else Fore.WHITE
        process_highlight = Style.BRIGHT if contains_process else Style.NORMAL

        if not contains_process: 
            str_results.append(f"{process_foreground}{process_highlight}   {star} {p.value} {Style.RESET_ALL}")

        else:  
            process_state_shortinfo = []
            process_state_longinfo = []

            if p is recreat_process.reclassification:
                if self.get_processing_parameter(p, recreat_process_parameters.export_name) is not None:
                    process_state_shortinfo.append(f"export={self.get_processing_parameter(p, recreat_process_parameters.export_name)}")
                for k,v in self.processes[p]['classes'].items():
                    process_state_longinfo.append(f"     {v} -> {k}")

            if p is recreat_process.clump_detection:
                process_state_shortinfo.append(f"barrier classes={self.get_processing_parameter(p, recreat_process_parameters.classes_on_restriction)}")

            if p is recreat_process.edge_detection:
                process_state_shortinfo.extend(
                    (
                        f"ignore edges to class={self.get_processing_parameter(p, recreat_process_parameters.classes_on_restriction)}",
                        "buffer=" + ','.join(map(str, self.buffered_edge_classes)),
                    )
                )
            if p is recreat_process.class_total_supply:
                process_state_shortinfo.append(
                    f"mode={self.get_processing_parameter(p, recreat_process_parameters.mode)}"
                )

            if p is recreat_process.aggregate_class_total_supply:                                
                if self.get_processing_parameter(p, recreat_process_parameters.export_non_weighted_results):
                    process_state_shortinfo.append("export non-weighted results")       
                process_state_longinfo.append(f"{process_foreground}      land-use weights={self.get_processing_parameter(p, recreat_process_parameters.lu_weights)}{Style.RESET_ALL}")
                                       

            if p is recreat_process.average_total_supply_across_cost:                                
                if self.get_processing_parameter(p, recreat_process_parameters.export_non_weighted_results):
                    process_state_shortinfo.append("export non-weighted results")                
                if self.get_processing_parameter(p, recreat_process_parameters.export_scaled_results):
                    process_state_shortinfo.append("export scaled results")   

                process_state_longinfo.append(f"{process_foreground}      cost weights={self.get_processing_parameter(p, recreat_process_parameters.cost_weights)}{Style.RESET_ALL}")
                process_state_longinfo.append(f"{process_foreground}      land-use weights={self.get_processing_parameter(p, recreat_process_parameters.lu_weights)}{Style.RESET_ALL}")
             

            if p is recreat_process.average_diversity_across_cost:                
                if self.get_processing_parameter(p, recreat_process_parameters.export_non_weighted_results):
                    process_state_shortinfo.append("export non-weighted results")                
                if self.get_processing_parameter(p, recreat_process_parameters.export_scaled_results):
                    process_state_shortinfo.append("export scaled results")     

                process_state_longinfo.append(f"{process_foreground}     cost weights={self.get_processing_parameter(p, recreat_process_parameters.cost_weights)}{Style.RESET_ALL}")
                           

            if p is recreat_process.proximity:
                process_state_shortinfo.append(f"mode={self.get_processing_parameter(p, recreat_process_parameters.mode)}")
                if self.get_processing_parameter(p, recreat_process_parameters.include_special_class):
                    process_state_shortinfo.append("assess proximities to built-up")

                process_state_longinfo.append(f"{process_foreground}      land-use classes={self.get_processing_parameter(p, recreat_process_parameters.classes_on_restriction)}{Style.RESET_ALL}")


            if p is recreat_process.average_cost:                                            
                if self.get_processing_parameter(p, recreat_process_parameters.user_threshold) > 0:
                    process_state_shortinfo.append(f"distance threshold={self.get_processing_parameter(p, recreat_process_parameters.user_threshold)}")                
                else:
                    process_state_shortinfo.append("no maximum distance masking")

                if self.get_processing_parameter(p, recreat_process_parameters.export_scaled_results):                             
                    process_state_shortinfo.append("export scaled results")

            if p is recreat_process.population_disaggregation:
                process_state_shortinfo.append(
                    f"population={self.get_processing_parameter(p, recreat_process_parameters.population_raster)}"
                )
                if self.get_processing_parameter(p, recreat_process_parameters.force):
                    process_state_shortinfo.append("force")
                if self.get_processing_parameter(p, recreat_process_parameters.export_scaled_results):
                    process_state_shortinfo("export scaled results")

            str_results.append(f"{process_foreground}{process_highlight}   {star} {p.value}{Style.RESET_ALL} {'(' if process_state_shortinfo else ''}{', '.join(process_state_shortinfo)}{')' if process_state_shortinfo else ''}")
            str_results += process_state_longinfo 

        return str_results


    def get_processes(self):
        return self.processes
    def get_model_parameters(self):
        return self.params
    def get_processing_parameter(self, process, param) -> any:
        if process in self.processes.keys() and param in self.processes[process].keys():
            return self.processes[process][param]
        else:
            return None

    def has_process_attached(self, process) -> bool:
        if process in self.processes.keys():
            return True
        else:
            return False

    def _add_key(self, target_dict, key, value):
        if not key in target_dict.keys():
            target_dict[key] = None
        target_dict[key] = value

    def _add_process(self, process):
        if not process in self.processes.keys():
            self.processes[process] = {}

            if process is recreat_process.reclassification:
                self.processes[process]['classes'] = {}

    def _add_process_config(self, process, process_param, config_value):
        self.processes[process][process_param] = config_value




