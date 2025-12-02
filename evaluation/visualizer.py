import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def plot_pareto_fronts(self, results_dict):
        """
        results_dict: { 'NSGA-II': [(m, c, e, r), ...], 'MOPSO': ... }
        """
        plt.figure(figsize=(10, 6))
        
        for algo, front in results_dict.items():
            # Extract Makespan vs Cost (Indices 0 and 1)
            makespans = [sol[0] for sol in front]
            costs = [sol[1] for sol in front]
            plt.scatter(makespans, costs, label=algo)
            
        plt.xlabel('Makespan')
        plt.ylabel('Cost')
        plt.title('Pareto Front Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'pareto_comparison.png'))
        plt.close()

    def plot_boxplots(self, df):
        """
        df: DataFrame with columns [Algorithm, Makespan, Cost, Energy, Reliability]
        """
        metrics = ['Makespan', 'Cost', 'Energy', 'Reliability']
        
        for metric in metrics:
            if metric in df.columns:
                plt.figure(figsize=(8, 5))
                sns.boxplot(x='Algorithm', y=metric, data=df)
                plt.title(f'{metric} Distribution')
                plt.savefig(os.path.join(self.output_dir, f'boxplot_{metric}.png'))
                plt.close()
