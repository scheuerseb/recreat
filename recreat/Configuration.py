###############################################################################
# (C) 2024 Sebastian Scheuer (seb.scheuer@outlook.de)                         #
###############################################################################

from enum import Enum
from typing import Dict

class Configuration:

    _task_type = None
    _args = None

    def __init__(self, current_task_type: any):
        self._args = {}
        self._task_type = current_task_type

    def add_arg(self, arg_name, arg_value):
        if arg_name not in self._args.keys():
            self._args[arg_name] = arg_value
    
    @property
    def args(self) -> Dict[str,any]:
        return self._args
    
    @property
    def task_type(self) -> any:
        return self._task_type
    