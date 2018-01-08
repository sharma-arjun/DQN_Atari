#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import numpy as np
import tensorflow as tf

# Importing necessary header files; using TensorFlow. 
import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss

from deeprl_hw2.core import *
from deeprl_hw2.policy import *

import gym

def get_output_folder(parent_dir, env_name):
        """Return save folder.

        Assumes folders in the parent_dir have suffix -run{run
        number}. Finds the highest run number and sets the output folder
        to that number + 1. This is just convenient so that if you run the
        same script multiple times tensorboard can plot all of the results
        on the same plots with different names.

        Parameters
        ----------
        parent_dir: str
          Path of the directory containing all experiment runs.

        Returns
        -------
        parent_dir/run_dir
          Path to this run's save directory.
        """
        os.makedirs(parent_dir, exist_ok=True)
        experiment_id = 0
        for folder_name in os.listdir(parent_dir):
                if not os.path.isdir(os.path.join(parent_dir, folder_name)):
                        continue
                try:
                        folder_name = int(folder_name.split('-run')[-1])
                        if folder_name > experiment_id:
                                experiment_id = folder_name
                except:
                        pass
        experiment_id += 1

        parent_dir = os.path.join(parent_dir, env_name)
        parent_dir = parent_dir + '-run{}'.format(experiment_id)
        return parent_dir


def main():  # noqa: D103

        parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
        parser.add_argument('--env', default='Enduro-v0', help='Atari env name')
        parser.add_argument('--seed', default=0, type=int, help='Random seed')
        parser.add_argument('--model_type', default='dqn', help='Model type: linear, dqn, double_linear, double_dqn')
        parser.add_arguement('--mode', default='train', help='Mode: train for training, test for testing')
        parser.add_arguement('--memory_size', default=200000, type=int, help='Replay memory size')
        parser.add_arguement('--save_every', default=50000, type=int, help='Frequency for saving weights')
        parser.add_arguement('--max_ep_length', default=50000, type=int, help='Maximum episode length during training')
        parser.add_arguement('--use_target_fixing', action='store_true', help='Use target fixing')
        parser.add_arguement('--use_replay_memory', action='store_true', help='Use replay memory')

        args = parser.parse_args()
        
        # Loading the appropriate environment. 
        env = gym.make('Enduro-v0')

        window = 4
        input_shape = (84,84)
        num_actions = env.action_space.n
                
        # Limit GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Set mode
        mode = args.mode

        # Set model variables

        # Model type to train.
        model_type = args.model_type

        # Initialize the Preprocessor, Memory, policy for training, 
        preproc = Preprocessor()
        memory = ReplayMemory(args.memory_size)
        policy = LinearDecayGreedyEpsilonPolicy(1,0.1,1000000, num_actions) # decay epsilon from 1 to 0.1 over 1 million steps

        # Setting experimental parameters - details of choices specified in the write up.
        gamma = 0.99
        target_update_freq = 10000
        num_burn_in = 1000
        train_freq = 0 # not using this parameter
        batch_size = 32
        target_fix_flag = args.target_fixing
        replay_mem_flag = args.replay_memory
        save_every = args.save_every


        print(sess)

        # Create a DQN agent with the specified parameters. 
        dqn = DQNAgent(sess, window, input_shape, num_actions, model_type, preproc, memory, policy, 
                                        gamma, target_fix_flag, target_update_freq, replay_mem_flag, num_burn_in, train_freq, batch_size, save_every)

        # Train the model on 3-5 Million frames, with given maximum episode length.
        if mode == 'train':
                dqn.fit(env, 5000000, args.max_ep_length)

        elif mode == 'test':

                # Load the model for testing. 
                model_file = 'saved_models_dqn/model_100000.ckpt'
                dqn.restore_model(model_file)

                # Evaluate the model.
                dqn.evaluate(env, 20 ,5000, 'test', lambda x: True, False, True)

if __name__ == '__main__':
        main()
