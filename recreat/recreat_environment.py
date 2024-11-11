from enum import Enum
from numpy import int32, float32, double
from typing import List, Dict
from colorama import Fore, Back, Style


class recreat_process(Enum):
    aggregation = 'aggregation'
    clump_detection = 'clumps'
    mask_landuses = 'mask-landuses'
    class_total_supply = 'class-total-supply'
    average_total_supply_across_cost = 'average-total-supply-across-cost'
    class_diversity = 'class-diversity'
    average_diversity_across_cost = 'average-diversity-across-cost'
    class_flow = 'class-flow'

class recreat_params(Enum):
    data_type = 'use-data-type'
    verbose = 'verbose-reporting'
    classes_patch = 'classes.patch'
    classes_edge = 'classes.edge'
    classes_builtup = 'classes.builtup'
    costs = 'costs'

class recreat_environment(Enum):
    data_path = 'working-directory'
    root = 'root-directory'
    landuse_file = 'land-use-file'
    nodata_values = 'land-use-file-nodata-values'
    nodata_fill_value = 'nodata-fill-value'

class recreat_processparams(Enum):
    lu_weights = 'landuse-weights'
    mode = 'mode'
    barrier_classes = 'barrier-classes'
    cost_weights = 'cost-weights'
    export_non_weighted_results = 'export-non-weighted-results'
    export_scaled_results = 'export-scaled-results'



class recreat_model():

    params = None   
    processes = None 
    
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


    # data path and root path
    @property
    def data_path(self) -> str:
        return self.environment[recreat_environment.data_path]
    @data_path.setter
    def data_path(self, data_path: str) -> None:
        self._add_key(self.environment, recreat_environment.data_path, data_path)
    @property
    def root_path(self) -> str:
        return self.environment[recreat_environment.root]
    @root_path.setter
    def root_path(self, root_dir: str) -> None:
        self._add_key(self.environment, recreat_environment.root, root_dir)

    # landuse file
    @property
    def landuse_file(self) -> str:
        return self.environment[recreat_environment.landuse_file]
    @landuse_file.setter
    def landuse_file(self, filename: str) -> None:
        self._add_key(self.environment, recreat_environment.landuse_file, filename)
    @property
    def landuse_file_nodata_values(self) -> List[float]:
        return self.environment[recreat_environment.nodata_values]
    @landuse_file_nodata_values.setter
    def landuse_file_nodata_values(self, nodata_vals: List[float]) -> None:
        self._add_key(self.environment, recreat_environment.nodata_values, nodata_vals)

    # verbosity, datatype
    @property
    def verbose(self) -> bool:
        return self.params[recreat_params.verbose]
    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._add_key(self.params, recreat_params.verbose, value)
    @property
    def datatype(self) -> str:
        return self.params[recreat_params.data_type]
    @datatype.setter
    def datatype(self, value: str) -> None:
        self._add_key(self.params, recreat_params.data_type, value)
    def datatype_as_numpy(self) -> any:
        pass
    
    # patch, edge, builtup classes
    @property
    def classes_patch(self) -> List[int]:
        return self.params[recreat_params.classes_patch]
    @classes_patch.setter
    def classes_patch(self, class_values: List[int]):
        self._add_key(self.params, recreat_params.classes_patch, class_values)
    @property
    def classes_edge(self) -> List[int]:
        return self.params[recreat_params.classes_edge]
    @classes_edge.setter
    def classes_edge(self, class_values: List[int]):
        self._add_key(self.params, recreat_params.classes_edge, class_values)
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
    
    # aggregations
    def add_aggregation(self, dest_class: int, source_classes: List[int]) -> None:
        self._add_process(recreat_process.aggregation)
        self.processes[recreat_process.aggregation][dest_class] = source_classes
    @property
    def aggregations(self) -> Dict[int, List[int]]:
        return self.processes[recreat_process.aggregation]
    
    # clump detection
    def add_clump_detection(self, barrier_classes: List[int]) -> None:
        self._add_process(recreat_process.clump_detection)
        self.processes[recreat_process.clump_detection][recreat_processparams.barrier_classes] = barrier_classes

    # landuse masking
    def add_mask_landuses(self) -> None:
        self._add_process(recreat_process.mask_landuses)

    # class total supply
    def add_class_total_supply(self, mode: str) -> None:
        self._add_process(recreat_process.class_total_supply)
        self.processes[recreat_process.class_total_supply][recreat_processparams.mode] = mode

    # average total supply across cost
    def add_average_total_supply_across_cost(self, lu_weights:Dict[int,float], cost_weights: Dict[int,float], export_non_weighted: bool, export_scaled: bool) -> None:
        self._add_process(recreat_process.average_total_supply_across_cost)
        self.processes[recreat_process.average_total_supply_across_cost][recreat_processparams.lu_weights] = lu_weights
        self.processes[recreat_process.average_total_supply_across_cost][recreat_processparams.cost_weights] = cost_weights
        self.processes[recreat_process.average_total_supply_across_cost][recreat_processparams.export_non_weighted_results] = export_non_weighted
        self.processes[recreat_process.average_total_supply_across_cost][recreat_processparams.export_scaled_results] = export_scaled

    # class diversity
    def add_class_diversity(self) -> None:
        self._add_process(recreat_process.class_diversity)

    # average diversity across cost
    def add_average_diversity_across_cost(self, cost_weights: Dict[int, float], export_non_weighted: bool, export_scaled: bool) -> None:
        self._add_process(recreat_process.average_diversity_across_cost)
        self.processes[recreat_process.average_diversity_across_cost][recreat_processparams.cost_weights] = cost_weights
        self.processes[recreat_process.average_diversity_across_cost][recreat_processparams.export_non_weighted_results] = export_non_weighted
        self.processes[recreat_process.average_diversity_across_cost][recreat_processparams.export_scaled_results] = export_scaled

    # class flow
    def add_class_flow(self) -> None:
        self._add_process(recreat_process.class_flow)






    def get_model_confirmation(self) -> bool:
        # print a summary
        print("Working directory: {}".format(self.data_path))
        print("Land-use data: {} from {}".format(self.landuse_file, self.root_path))
        print()
        print("Patches: {}".format(self.classes_patch))
        print("Edges: {}".format(self.classes_edge))
        print("Built-up: {}".format(self.classes_builtup))
        print("Costs: {}".format(self.costs))

        print()
        print("Process:")
        for p in recreat_process:
            pars = ""            
            contains_process = True if p in self.processes.keys() else False
            
            star = ' ' if not contains_process else '*'
            fore = Fore.YELLOW if contains_process else Fore.WHITE
            highlight = Style.BRIGHT if contains_process else Style.NORMAL
            
            # process items to be shown as (pars) with title 

            if p is recreat_process.clump_detection and contains_process:
                pars = "Barriers={}".format(self.processes[p][recreat_processparams.barrier_classes])
            if p is recreat_process.class_total_supply and contains_process:
                pars = "Mode={}".format(self.processes[p][recreat_processparams.mode])
            if p is recreat_process.average_total_supply_across_cost and contains_process:                
                vals = []
                if self.processes[p][recreat_processparams.export_non_weighted_results]:
                    vals.append("Export non-weighted results")                
                if self.processes[p][recreat_processparams.export_scaled_results]:
                    vals.append("Export scaled results")                
                pars = ", ".join(vals)
            if p is recreat_process.average_diversity_across_cost and contains_process:                
                vals = []
                if self.processes[p][recreat_processparams.export_non_weighted_results]:
                    vals.append("Export non-weighted results")                
                if self.processes[p][recreat_processparams.export_scaled_results]:
                    vals.append("Export scaled results")                
                pars = ", ".join(vals)
            

            op_bracket = "" if pars == "" else "("
            cl_bracket = "" if pars == "" else ")"
            print(fore + highlight + "   {} {} {}{}{}".format(star, p.value, op_bracket, pars, cl_bracket) + Style.RESET_ALL)

            # process items to be shown in lines under title
            if p is recreat_process.aggregation and contains_process:
                for k,v in self.processes[p].items():
                    print(fore + "     {} -> {}".format(v, k) + Style.RESET_ALL)

            if p is recreat_process.average_total_supply_across_cost and contains_process:
                print(fore + "     {}={}".format('cost weights', self.processes[p][recreat_processparams.cost_weights]) + Style.RESET_ALL)
                print(fore + "     {}={}".format('landuse weights', self.processes[p][recreat_processparams.lu_weights]) + Style.RESET_ALL)
            if p is recreat_process.average_diversity_across_cost and contains_process:
                print(fore + "     {}={}".format('cost weights', self.processes[p][recreat_processparams.cost_weights]) + Style.RESET_ALL)



        print()
        return True
        #user_confirm = input("Run this model? (Y/n): ")
        #if user_confirm is None or user_confirm == '' or user_confirm.lower() == 'y':
        #    return True
        #else:
        #    return False
          

    def get_processes(self):
        return recreat_process


    def _add_key(self, target_dict, key, value):
        if not key in target_dict.keys():
            target_dict[key] = None
        target_dict[key] = value

    def _add_process(self, process):
        if not process in self.processes.keys():
            self.processes[process] = {}




