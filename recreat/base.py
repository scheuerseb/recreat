from colorama import init as colorama_init
from colorama import Fore, Back, Style, just_fix_windows_console
from rich.progress import Progress, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn, BarColumn

class RecreatBase:
    # environment variables
    data_path = None       # path to datasets
    root_path = None       # path to a specific "scenario" to be assessed, i.e., subfolder in data_path
    
    # progress reporting
    progress = None     

    def __init__(self, data_path: str, root_path: str) -> None:
        self.data_path = data_path
        self.root_path = root_path
    
    def printStepInfo(self, msg):
        print(Fore.CYAN + Style.BRIGHT + msg.upper() + Style.RESET_ALL)
    
    def printStepCompleteInfo(self, msg = "COMPLETED"):
        print(Fore.GREEN + Style.BRIGHT + msg + Style.RESET_ALL)

    def taskProgressReportStepCompleted(self, msg = "COMPLETED"):
        self.printStepCompleteInfo(msg = msg)

    def _new_progress(self, task_description, total):
        self.progress = self.get_progress_bar()
        return self.progress.add_task(f"{task_description:<40}", total=total)

    def get_progress_bar(self):
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn()
        )
    
    def _get_task(self, task_description, total):
        return self._new_progress(task_description, total=total)
