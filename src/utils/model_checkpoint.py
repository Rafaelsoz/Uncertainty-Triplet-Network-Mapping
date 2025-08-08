import pickle
from typing import Callable


class Checkpoint:
    def __init__(self, args, metric_name: str, initial_value: float, output_dir: str):
        self.args = args
        self.metric_name = metric_name
        self.value = initial_value
        self.output_dr = output_dir

        self.save_dict = None
    
    def __call__(
        self, 
        model: Callable,
        stats: dict,
    ):
        self.save_dict = {
            'stats': stats,
            'args': self.args,
            'model_state':model.state_dict()
        }

        self.value = stats[self.metric_name]

    def save(self, file_name: str):
        with open(f'{file_name}.pkl', 'wb') as f:
            pickle.dump(self.save_dict, f)
    
    def open(self, file_name: str):
        with open(f'{file_name}.pkl', 'rb') as f:
            self.save_dict = pickle.load(f)
        
