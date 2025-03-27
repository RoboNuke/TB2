import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

args_cli.video = False
args_cli.enable_cameras = False

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import torch.multiprocessing as mp
from agents.wandb_logger_ppo_agent import WandbLoggerPPO
from models.bro_model import BroAgent
import gymnasium
import copy
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from multiprocessing.managers import BaseManager

#class MyManager(BaseManager):
#    pass

#MyManager.register("agent", WandbLoggerPPO)

def thread_func(process_index, *args):
    print(f"[INFO] Processor {process_index}: started")

    pipe = args[0][process_index]
    queue = args[1][process_index]
    barrier = args[2]


    agent = queue.get()
    #print("got agent")
    #time.sleep(10)
    agent.init(trainer_cfg=queue.get())
    #print(f"[INFO] Processor {process_index}: init agent {type(agent).__name__}")
    barrier.wait()
    
    print(f"{process_index} has reached the barrier")

    #time.sleep(2 * (process_index+1))
    # wait for the main process to start all the workers
    barrier.wait()
    print(f"{process_index} has ended")

if __name__ == "__main__":

    mp.set_start_method("spawn")
    n_threads = 2
    queues = []
    producer_pipes = []
    consumer_pipes = []
    barrier = mp.Barrier(n_threads + 1)
    processes = []

    #print("Vars init")
    model = {}
    model['policy'] = BroAgent(
        observation_space=gymnasium.spaces.Box(0, 1, shape=(2,)), 
        action_space=gymnasium.spaces.Box(0, 1, shape=(2,)),
        device = "cuda:0"
    ) 
    model["value"] = model["policy"]  # same instance: shared model

    models = [copy.deepcopy(model) for i in range(n_threads)]

    for model in models:
        model['policy'].share_memory()
        model["value"].share_memory()
    #print("Model init")

    agent_list = [
        WandbLoggerPPO(
            models=models[i]#,
            #cfg={"is_distributed": True}
        ) for i in range(n_threads)
    ]
    """
    manager = MyManager()
    manager.start()
    agent_list = [
        manager.agent(
            models=models[i]#,
            #cfg={"is_distributed": True}
        ) for i in range(n_threads)
    ]
    """
    #for agent in agent_list:
    #    agent.share_memory()
    print("Agents init")


    for i in range(n_threads):
        pipe_read, pipe_write = mp.Pipe(duplex=False)
        producer_pipes.append(pipe_write)
        consumer_pipes.append(pipe_read)
        queues.append(mp.Queue())


    print("Thread stuff init")
    # spawn and wait for all processes to start
    for i in range(n_threads):
        queues[i].put(agent_list[i])
        queues[i].put({}) #"is_distributed":True
        process = mp.Process(
            target=thread_func, 
            args= (
                i, 
                consumer_pipes, 
                queues, 
                barrier
            ), daemon=True
        )
        processes.append(process)
        process.start()
    

    print("Main waiting at barrier")
    barrier.wait()
    for process in processes:
        print("joining proci")
        process.join()
    print("Program exiting")
    
    
