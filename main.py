import argparse
import os
from config.manager import ConfigManager
from utils.output_manager import OutputManager
from workflows.generator import WorkflowGenerator
from workflows.parser import WorkflowParser
from simulator.resource import CloudNode, EdgeNode
from algorithms.heft import HEFT
from algorithms.nsga2 import NSGA2Scheduler
# from algorithms.mopso import MOPSOScheduler
# from algorithms.paes import PAESScheduler

def setup_resources(config):
    cloud_cfg = config.get('resources.cloud_nodes')
    edge_cfg = config.get('resources.edge_nodes')
    
    cloud_nodes = [CloudNode(i, cloud_cfg) for i in range(cloud_cfg['count'])]
    edge_nodes = [EdgeNode(i + cloud_cfg['count'], edge_cfg) for i in range(edge_cfg['count'])]
    
    return cloud_nodes + edge_nodes

def main():
    parser = argparse.ArgumentParser(description="Multi-Objective Workflow Scheduling")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # 1. Load Config & Setup Output
    cfg = ConfigManager(args.config)
    out = OutputManager(cfg.get('logging.save_dir'))
    out.log(f"Loaded configuration from {args.config}")

    # 2. Load/Generate Workflow
    wf_cfg = cfg.get('workflow')
    wf_source = wf_cfg.get('source', 'synthetic')
    wf_type = wf_cfg.get('type', 'random')
    wf_size = wf_cfg.get('size', 50)
    
    if wf_source == 'file' and wf_cfg.get('file_path'):
        # Load from existing file
        out.log(f"Loading workflow from file: {wf_cfg['file_path']}...")
        parser = WorkflowParser()
        tasks, graph = parser.parse_file(wf_cfg['file_path'])
    elif wf_source == 'file' or wf_type.startswith('pegasus_'):
        # Generate real Pegasus workflow dynamically
        actual_type = wf_type.replace('pegasus_', '') if wf_type.startswith('pegasus_') else wf_type
        out.log(f"Generating real Pegasus {actual_type} workflow (Size={wf_size})...")
        
        from workflows.pegasus_generator import PegasusWorkflowGenerator
        import tempfile
        import os
        
        # Generate to temp file then parse
        temp_file = os.path.join(tempfile.gettempdir(), f"workflow_{actual_type}_{wf_size}.yml")
        try:
            PegasusWorkflowGenerator.generate_workflow(actual_type, wf_size, temp_file)
            parser = WorkflowParser()
            tasks, graph = parser.parse_file(temp_file)
        except Exception as e:
            out.log(f"ERROR: Failed to generate Pegasus workflow: {e}. Falling back to synthetic.")
            generator = WorkflowGenerator()
            tasks, graph = generator.generate_workflow(actual_type, wf_size)
    else:
        # Generate synthetic workflow
        out.log(f"Generating synthetic {wf_type} workflow (Size={wf_size})...")
        generator = WorkflowGenerator()
        tasks, graph = generator.generate_workflow(wf_type, wf_size)
    
    out.log(f"Workflow has {len(tasks)} tasks.")

    # 3. Setup Resources
    nodes = setup_resources(cfg)
    out.log(f"Setup {len(nodes)} resources.")

    # 4. Run Algorithms
    algo_name = cfg.get('algorithm.name')
    results_dict = {}
    
    if algo_name in ["heft", "all"]:
        out.log("Running HEFT...")
        heft = HEFT(tasks, graph, nodes)
        schedule = heft.schedule()
        
        # Calculate full metrics using Environment
        from simulator.environment import CloudEdgeEnvironment
        env = CloudEdgeEnvironment(nodes[:cfg.get('resources.cloud_nodes.count')], 
                                   nodes[cfg.get('resources.cloud_nodes.count'):],
                                   bandwidth_cloud_edge=cfg.get('simulation.bandwidth_cloud_edge'))
        
        # Convert HEFT schedule (list of (task, node)) to env format
        # HEFT schedule objects might need mapping. 
        # Assuming HEFT returns list of (Task, Node) objects.
        # Environment.run expects (Task, node_id)
        env_schedule = []
        for task, node_id in schedule:
            env_schedule.append((task, node_id))
            
        metrics = env.run(env_schedule)
        
        heft_res = [(metrics['makespan'], metrics['cost'], metrics['energy'], metrics['reliability'], metrics['load_balancing'])]
        results_dict['HEFT'] = heft_res
        out.log(f"HEFT completed. Schedule length: {len(schedule)}")

    if algo_name in ["nsga2", "all"]:
        out.log("Running NSGA-II...")
        nsga2 = NSGA2Scheduler(tasks, graph, nodes, 
                               population_size=cfg.get('algorithm.population_size'),
                               generations=cfg.get('algorithm.generations'))
        pareto_front = nsga2.run()
        # Extract fitness values
        front_fits = [ind.fitness.values for ind in pareto_front]
        results_dict['NSGA-II'] = front_fits
        out.log(f"NSGA-II completed. Pareto front size: {len(pareto_front)}")

    if algo_name in ["mopso", "all"]:
        from algorithms.mopso import MOPSOScheduler
        out.log("Running MOPSO...")
        mopso = MOPSOScheduler(tasks, graph, nodes,
                               population_size=cfg.get('algorithm.population_size'),
                               generations=cfg.get('algorithm.generations'))
        archive = mopso.run()
        # Extract fitness values
        archive_fits = [p['fitness'] for p in archive]
        results_dict['MOPSO'] = archive_fits
        out.log(f"MOPSO completed. Archive size: {len(archive)}")
        
    if algo_name in ["paes", "all"]:
        from algorithms.paes import PAESScheduler
        out.log("Running PAES...")
        paes = PAESScheduler(tasks, graph, nodes)
        archive = paes.run()
        # Extract fitness values
        archive_fits = [p['fitness'] for p in archive]
        results_dict['PAES'] = archive_fits
        out.log(f"PAES completed. Archive size: {len(archive)}")
    
    if algo_name in ["mor2ga", "all"]:
        from algorithms.mor2ga import MOR2GA
        out.log("Running MO-R²GA...")
        mor2ga = MOR2GA(tasks, graph, nodes,
                        pop_size=cfg.get('algorithm.population_size'),
                        generations=cfg.get('algorithm.generations'))
        pareto_front = mor2ga.run()
        
        # Extract fitness values (individuals already have .fitness.values)
        mor2ga_fits = [ind.fitness.values for ind in pareto_front]
        
        results_dict['MO-R2GA'] = mor2ga_fits
        out.log(f"MO-R²GA completed. Pareto front size: {len(mor2ga_fits)}")

    if algo_name in ["smartga", "all"]:
        from algorithms.smartga import SmartGA
        out.log("Running SmartGA (NSGA-III placeholder)...")
        smartga = SmartGA(tasks, graph, nodes,
                         pop_size=cfg.get('algorithm.population_size'),
                         generations=50)  # Increased to 50 for better convergence
        pareto_schedules = smartga.schedule()
        
        # Extract fitness values from stored objectives
        smartga_fits = []
        for obj_values in smartga.pareto_front_objectives:
            # obj_values = [makespan, cost, load_bal]
            smartga_fits.append(tuple(obj_values))
        
        results_dict['SmartGA'] = smartga_fits
        out.log(f"SmartGA completed. Pareto front size: {len(smartga_fits)}")


    # 5. Visualization & Saving
    if results_dict:
        # Save results to OutputManager
        for algo, res_list in results_dict.items():
            for res in res_list:
                # res is a tuple/list of fitness values
                # Case 1: 5 Objectives (Old/HEFT) -> (Makespan, Cost, Energy, Reliability, LoadBalancing)
                # Case 2: 3 Objectives (New) -> (Makespan, Cost, LoadBalancing)
                
                metrics = {}
                if len(res) == 3:
                    metrics = {
                        "makespan": res[0],
                        "cost": res[1],
                        "energy": 0.0, # Not optimized
                        "reliability": 0.0, # Not optimized
                        "load_balancing": res[2],
                        "workflow_size": cfg.get('workflow.size')
                    }
                elif len(res) >= 5:
                    metrics = {
                        "makespan": res[0],
                        "cost": res[1],
                        "energy": res[2],
                        "reliability": abs(res[3]), # Handle potential negative
                        "load_balancing": res[4],
                        "workflow_size": cfg.get('workflow.size')
                    }
                
                out.save_result(algo, None, metrics)

        from evaluation.visualizer import Visualizer
        vis = Visualizer(out.exp_dir)
        vis.plot_pareto_fronts(results_dict)
        out.log("Generated Pareto front comparison plot.")

    # Export results
    csv_path = out.export_csv()
    print(f"Experiment completed. Results saved to {csv_path}")

if __name__ == "__main__":
    main()
