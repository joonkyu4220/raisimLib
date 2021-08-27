if __name__ == '__main__':
   import json
   from ruamel.yaml import YAML, dump, RoundTripDumper
   from raisimGymTorch.env.bin import solo8_env
   from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecTorchEnv as VecEnv
   from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
 
   import torch
   import torch.optim as optim
   import torch.multiprocessing as mp
   import torch.nn as nn
   import torch.nn.functional as F
   from torch.autograd import Variable
   import torch.utils.data
   from model import ActorCriticNet, Shared_obs_stats
   import os
   import numpy as np
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
   seed = 1#8
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.set_num_threads(1)
 
   # directories
   task_path = os.path.dirname(os.path.realpath(__file__))
   home_path = task_path + "/../../../../.."

   # config
   cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

   # create environment from the configuration file
   env = VecEnv(solo8_env.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
   print("env_created")

   num_inputs = env.observation_space.shape[0]
   num_outputs = env.action_space.shape[0]
   model = ActorCriticNet(num_inputs, num_outputs, [128, 128])
   model_v2 = ActorCriticNet(num_inputs, num_outputs, [128, 128])
   # model.load_state_dict(torch.load("stats/solo8_July21_symmetry/iter9400.pt"))
   # model.load_state_dict(torch.load("stats/solo8_Aug02_flip/iter3800.pt"))
   # model_v2.load_state_dict(torch.load("stats/solo8_Aug11_flip_v2/iter2700.pt"))
   # model.load_state_dict(torch.load("stats/solo8_Aug17_flip_AAB//iter3200.pt"))
   model.load_state_dict(torch.load("stats/solo8_Aug26_flip_DDD//iter9900.pt"))
   model.cuda()
   model_v2.cuda()
   model.set_noise(-2.5 * np.ones(num_outputs))

   env.reset()
   obs = env.observe()
   for i in range(10000000):
      with torch.no_grad():
         act = model.sample_best_actions(obs + torch.randn_like(obs).mul(0.1))
         # act[:, 1] = 0.2 
      # act = torch.round(act * 10^3) / (10^3)
      # act = (act * 100).round() / 100
      obs, rew, done, _ = env.step(act)
      env.reset_time_limit() 

      import time; time.sleep(0.01)
      # if i % 60 == 0:
      #    env.reset()
      # print(rew, done)