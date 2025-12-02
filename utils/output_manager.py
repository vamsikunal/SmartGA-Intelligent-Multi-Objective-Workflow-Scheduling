import os
import json
import logging
import pandas as pd
from datetime import datetime

class OutputManager:
    def __init__(self, base_dir="results"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.logger = self._setup_logger()
        self.results = []

    def _setup_logger(self):
        logger = logging.getLogger("WorkflowScheduler")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.exp_dir, "experiment.log"))
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def log(self, message):
        self.logger.info(message)

    def save_result(self, algorithm, solution, metrics):
        """
        Saves a single solution result.
        metrics: dict of {makespan, cost, energy, reliability}
        """
        record = {
            "algorithm": algorithm,
            **metrics
        }
        self.results.append(record)

    def export_csv(self):
        df = pd.DataFrame(self.results)
        path = os.path.join(self.exp_dir, "all_results.csv")
        df.to_csv(path, index=False)
        self.log(f"Results saved to {path}")
        return path
