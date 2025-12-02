"""
Dynamic Pegasus Workflow Generator
Generates real Pegasus YAML workflows with configurable sizes.
"""

from Pegasus.api import *

class PegasusWorkflowGenerator:
    """Generate real Pegasus workflow definitions dynamically."""
    
    @staticmethod
    def generate_diamond(size=4):
        """Generate a diamond workflow (4 tasks fixed)."""
        wf = Workflow("diamond")
        
        preprocess = Job("preprocess", _id="ID0000001")
        findrange_1 = Job("findrange", _id="ID0000002") 
        findrange_2 = Job("findrange", _id="ID0000003")
        analyze = Job("analyze", _id="ID0000004")
        
        wf.add_jobs(preprocess, findrange_1, findrange_2, analyze)
        wf.add_dependency(preprocess, children=[findrange_1, findrange_2])
        wf.add_dependency(findrange_1, children=[analyze])
        wf.add_dependency(findrange_2, children=[analyze])
        
        return wf
    
    @staticmethod
    def generate_montage(num_tasks=50):
        """
        Generate Montage workflow with configurable size.
        Structure: mProject -> mDiffFit -> mConcatFit -> mBgModel -> mBackground -> mAdd
        """
        wf = Workflow("montage")
        
        # Calculate layer sizes based on total tasks
        # Roughly: 40% project, 30% diff, 20% background, rest for aggregation
        num_images = max(2, int(num_tasks * 0.4))
        
        # Level 1: Project images
        project_jobs = []
        for i in range(num_images):
            job = Job("mProject", _id=f"mProject_{i}")
            project_jobs.append(job)
            wf.add_jobs(job)
        
        # Level 2: Difference fitting (pairwise)
        diff_jobs = []
        num_diffs = min(num_images - 1, max(1, int(num_tasks * 0.3)))
        for i in range(num_diffs):
            job = Job("mDiffFit", _id=f"mDiffFit_{i}")
            diff_jobs.append(job)
            wf.add_jobs(job)
            # Connect to adjacent project jobs
            if i < len(project_jobs):
                wf.add_dependency(project_jobs[i], children=[job])
            if i + 1 < len(project_jobs):
                wf.add_dependency(project_jobs[i+1], children=[job])
        
        # Level 3: Concat
        concat_job = Job("mConcatFit", _id="mConcatFit")
        wf.add_jobs(concat_job)
        for diff_job in diff_jobs:
            wf.add_dependency(diff_job, children=[concat_job])
        
        # Level 4: Background model
        bgmodel_job = Job("mBgModel", _id="mBgModel")
        wf.add_jobs(bgmodel_job)
        wf.add_dependency(concat_job, children=[bgmodel_job])
        
        # Level 5: Background correction
        background_jobs = []
        for i in range(num_images):
            job = Job("mBackground", _id=f"mBackground_{i}")
            background_jobs.append(job)
            wf.add_jobs(job)
            wf.add_dependency(project_jobs[i], children=[job])
            wf.add_dependency(bgmodel_job, children=[job])
        
        # Level 6: Final mosaic
        add_job = Job("mAdd", _id="mAdd")
        wf.add_jobs(add_job)
        for bg_job in background_jobs:
            wf.add_dependency(bg_job, children=[add_job])
        
        return wf
    
    @staticmethod
    def generate_cybershake(num_tasks=50):
        """
        Generate CyberShake workflow with configurable size.
        Structure: Parallel pipelines (Extract -> SGT -> Seismogram) -> PeakVals
        """
        wf = Workflow("cybershake")
        
        # Calculate number of pipelines
        pipeline_len = 3
        num_pipelines = max(1, (num_tasks - 1) // pipeline_len)
        
        final_node = Job("PeakVals", _id="peakvals")
        wf.add_jobs(final_node)
        
        for site in range(num_pipelines):
            extract = Job("ExtractSGT", _id=f"extract_{site}")
            sgt = Job("SeismogramSynthesis", _id=f"sgt_{site}")
            seismo = Job("Seismogram", _id=f"seismo_{site}")
            
            wf.add_jobs(extract, sgt, seismo)
            wf.add_dependency(extract, children=[sgt])
            wf.add_dependency(sgt, children=[seismo])
            wf.add_dependency(seismo, children=[final_node])
        
        return wf
    
    @staticmethod
    def generate_sipht(num_tasks=50):
        """
        Generate SIPHT workflow with configurable size.
        Structure: Source -> Parallel Processing -> Aggregation
        """
        wf = Workflow("sipht")
        
        # Source + Sink = 2 tasks, rest are parallel
        num_parallel = max(1, num_tasks - 2)
        
        source = Job("Preprocess", _id="preprocess")
        wf.add_jobs(source)
        
        process_jobs = []
        for i in range(num_parallel):
            job = Job("Analyze", _id=f"analyze_{i}")
            wf.add_jobs(job)
            wf.add_dependency(source, children=[job])
            process_jobs.append(job)
        
        aggregate = Job("Aggregate", _id="aggregate")
        wf.add_jobs(aggregate)
        for job in process_jobs:
            wf.add_dependency(job, children=[aggregate])
        
        return wf
    
    @classmethod
    def generate_workflow(cls, workflow_type, num_tasks, output_file=None):
        """
        Generate a Pegasus workflow of the specified type and size.
        
        Args:
            workflow_type: Type of workflow (diamond, montage, cybershake, sipht)
            num_tasks: Target number of tasks
            output_file: Optional file to write workflow to
            
        Returns:
            Pegasus Workflow object
        """
        generators = {
            'diamond': cls.generate_diamond,
            'montage': cls.generate_montage,
            'cybershake': cls.generate_cybershake,
            'sipht': cls.generate_sipht
        }
        
        if workflow_type not in generators:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        wf = generators[workflow_type](num_tasks)
        
        if output_file:
            wf.write(output_file)
        
        return wf
