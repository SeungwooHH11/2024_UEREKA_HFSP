import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools


def run_hybrid_flexible_flowshop(num_stages, machines_per_stage, processing_times, job_order):
    class Job:
        def __init__(self, job_id, processing_times):
            self.job_id = job_id
            self.processing_times = processing_times
            self.current_stage = 0
            self.completion_time = 0
            self.start_times = []
            self.end_times = []

    class Machine:
        def __init__(self, machine_id):
            self.machine_id = machine_id
            self.available_time = 0

    jobs = [Job(i, processing_times[i]) for i in range(len(processing_times))]
    machines = [[Machine(j) for j in range(machines_per_stage[i])] for i in range(num_stages)]

    while any(job.current_stage < num_stages for job in jobs):
        for stage in range(num_stages):
            for job_id in job_order:
                job = jobs[job_id]
                if job.current_stage == stage:
                    # Schedule the job
                    stage_machines = machines[stage]
                    machine = min(stage_machines, key=lambda x: x.available_time)
                    start_time = max(machine.available_time, job.completion_time)
                    end_time = start_time + job.processing_times[stage]

                    job.start_times.append(start_time)
                    job.end_times.append(end_time)

                    job.completion_time = end_time
                    machine.available_time = end_time
                    job.current_stage += 1

    makespan = max(job.completion_time for job in jobs)
    job_completion_times = [(job.job_id, job.completion_time) for job in jobs]
    return makespan, job_completion_times


def calculate_makespans(num_stages, machines_per_stage, processing_times_list, job_orders):
    makespans = []
    for processing_times, job_order in zip(processing_times_list, job_orders):
        makespan, _ = run_hybrid_flexible_flowshop(num_stages, machines_per_stage, processing_times, job_order)
        makespans.append(makespan)
    return  torch.tensor(np.array(makespans),dtype=torch.float32)



def plot_gantt_chart(jobs):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(jobs)))

    for i, job in enumerate(jobs):
        for stage in range(len(job.start_times)):
            ax.barh(job.job_id, job.end_times[stage] - job.start_times[stage], left=job.start_times[stage],
                    color=colors[i], edgecolor='black', alpha=0.7)
            ax.text((job.start_times[stage] + job.end_times[stage]) / 2, job.job_id, f'S{stage}', ha='center',
                    va='center', color='white', fontsize=8)

    ax.set_yticks([job.job_id for job in jobs])
    ax.set_yticklabels([f'Job {job.job_id}' for job in jobs])
    ax.set_xlabel('Time')
    ax.set_ylabel('Jobs')
    ax.set_title('Gantt Chart for Hybrid Flexible Flowshop')
    plt.show()


def generate_processing_times(num_problem, num_of_ep , num_jobs, num_stages, max_time):
    """
    Generate random processing times for each job at each stage across multiple batches.

    Parameters:
    num_batches (int): Number of batches
    num_jobs (int): Number of jobs per batch
    num_stages (int): Number of stages per job
    max_time (int): Maximum processing time

    Returns:
    list: A list of batches, where each batch is a list of lists containing random processing times for jobs
    """
    batches = []
    for _ in range(num_problem):
        processing_times = []
        for _ in range(num_jobs):
            job_times = [random.randint(0, max_time) for _ in range(num_stages)]
            processing_times.append(job_times)
        for _ in range(num_of_ep)
            batches.append(processing_times.copy())
    return torch.tensor(np.array(batches),dtype=torch.float32)
