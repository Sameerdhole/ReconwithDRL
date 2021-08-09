import sys, cv2
import nvidia_smi
from network.agent import PedraAgent
from unreal_envs.initial_positions import *
from os import getpid
from network.Memory import Memory
from aux_functions import *
import os
from util.transformations import euler_from_quaternion
from configs.read_cfg import read_cfg, update_algorithm_cfg
from DeepPPO import DeepPPO



def DeepPPG(cfg,env_process,env_folder):
	algorithm_cfg = read_cfg(config_filename='configs/DeepPPG.cfg', verbose=True)
    algorithm_cfg.algorithm = cfg.algorithm

    if 'GlobalLearningGlobalUpdate-SA' in algorithm_cfg.distributed_algo:

    
    if cfg.mode == 'train':

    elif cfg.mode == 'infer':
