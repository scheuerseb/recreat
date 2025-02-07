###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################

from colorama import init as colorama_init
from colorama import Fore, Back, Style, just_fix_windows_console
from rich.progress import Progress, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn, BarColumn
import numpy as np
import rasterio

class RecreatBase:
    # environment variables
    data_path = None       # path to datasets
    root_path = None       # path to a specific "scenario" to be assessed, i.e., subfolder in data_path
    
    # progress reporting
    progress = None     

    def __init__(self, data_path: str, root_path: str) -> None:
        self.data_path = data_path
        self.root_path = root_path
    
    # support static methods
    def printStepInfo(self, msg):
        RecreatBase.printStepInfo(msg.upper())
    @staticmethod 
    def printStepInfo(msg):
        print(Fore.CYAN + Style.BRIGHT + msg.upper() + Style.RESET_ALL)
    
    def printStepCompleteInfo(self, msg = "COMPLETED"):
        RecreatBase.printStepCompleteInfo(msg)
    @staticmethod
    def printStepCompleteInfo(msg = "COMPLETED"):
        print(Fore.GREEN + Style.BRIGHT + msg + Style.RESET_ALL)

    def taskProgressReportStepCompleted(self, msg = "COMPLETED"):
        RecreatBase.taskProgressReportStepCompleted(msg)
    @staticmethod
    def taskProgressReportStepCompleted(msg = "COMPLETED"):
        RecreatBase.printStepCompleteInfo(msg = msg)

    # task progress not supporting static methods at this time
    def _task_new(self, task_description, total):
        self.progress = self.get_progress_bar()
        return self._task_add(task_description, total)

    def _task_add(self, task_description, total):
        return self.progress.add_task(f"{task_description:<40}", total=total)

    def get_file_path(self, file_name: str, relative_to_root_path: bool = True):
        """Get the fully-qualified path to model file with specified filename.

        :param file_name: Model file for which the fully qualified path should be generated. 
        :type file_name: str
        :param is_scenario_specific: Indicates if the specified datasource located in a scenario-specific root-path (True) or at the data-path  (False), defaults to True.
        :type is_scenario_specific: bool, optional       
        """
        return (
            f"{self.data_path}/{file_name}"
            if not relative_to_root_path
            else f"{self.data_path}/{self.root_path}/{file_name}"
        )

    def get_progress_bar(self):
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn()
        )
    
    def _new_task(self, task_description, total):
        return self._task_new(task_description, total=total)
    def _new_subtask(self, task_desciption, total):
        return self._task_add(task_desciption, total=total)


    
    @staticmethod
    def write_output(out_filename: str, out_mtx: np.ndarray, out_meta) -> None:
        with rasterio.open(out_filename, "w", **out_meta) as dest:
            dest.write(out_mtx, 1)