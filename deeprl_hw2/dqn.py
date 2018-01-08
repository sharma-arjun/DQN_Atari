"""Main DQN agent."""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import copy
import os
import deeprl_hw2 as tfrl
from gym import wrappers
from deeprl_hw2.objectives import mean_huber_loss

from deeprl_hw2.core import *
from deeprl_hw2.utils import get_hard_target_model_updates
from deeprl_hw2.policy import GreedyEpsilonPolicy
import cProfile
# from core import Sample

class DQNAgent:
        """Class implementing DQN.

        This is a basic outline of the functions/parameters you will need
        in order to implement the DQNAgnet. This is just to get you
        started. You may need to tweak the parameters, add new ones, etc.

        Feel free to change the functions and funciton parameters that the
        class provides.

        We have provided docstrings to go along with our suggested API.

        Parameters
        ----------
        q_network: keras.models.Model
          Your Q-network model.
        preprocessor: deeprl_hw2.core.Preprocessor
          The preprocessor class. See the associated classes for more
          details.
        memory: deeprl_hw2.core.Memory
          Your replay memory.
        gamma: float
          Discount factor.
        target_update_freq: float
          Frequency to update the target network. You can either provide a
          number representing a soft target update (see utils.py) or a
          hard target update (see utils.py and Atari paper.)
        num_burn_in: int
          Before you begin updating the Q-network your replay memory has
          to be filled up with some number of samples. This number says
          how many.
        train_freq: int
          How often you actually update your Q-Network. Sometimes
          stability is improved if you collect a couple samples for your
          replay memory, for every Q-network update that you run.
        batch_size: int
          How many samples in each minibatch.
        """
        def __init__(self,sess,
                                 window,
                                 input_shape,
                                 num_actions,
                                 model_type,
                                 preprocessor,
                                 memory,
                                 policy,
                                 gamma,
                                 target_fix_flag,
                                 target_update_freq,
                                 replay_mem_flag,
                                 num_burn_in,
                                 train_freq,
                                 batch_size,
                                 save_every):

                # Defining core parameters as variables of the DQN agent class.
                self.sess = sess
                self.preprocessor = preprocessor
                self.memory = memory
                self.policy = policy
                self.gamma = gamma
                self.target_update_freq = target_update_freq
                self.num_burn_in = num_burn_in
                self.train_freq = train_freq
                self.batch_size = batch_size
                self.num_actions = num_actions
                self.window = window
                self.target_fix_flag = target_fix_flag
                self.replay_memory_flag = replay_mem_flag
                self.source_net = 'source_' + model_type

                # If using target fixing, also name a target model.
                if target_fix_flag:
                        self.target_net = 'target_' + model_type

                self.save_every = save_every
                self.average_reward = 0.0
                self.model_type = model_type
                self.is_restored = False

                # Based on model type, call appropriate function to create the model.
                if model_type == 'linear' or model_type == 'double_linear':
                        self.create_linear_model(window,input_shape,num_actions,self.source_net)
                        if target_fix_flag:
                                # Create another instance if the target fixing flag is fixed.
                                self.create_linear_model(window,input_shape,num_actions,self.target_net)

                # Corresponding code for DQN and Double DQN.
                elif model_type == 'dqn' or model_type == 'double_dqn':
                        self.create_dqn_model(window,input_shape,num_actions,self.source_net)
                        if target_fix_flag:
                                self.create_dqn_model(window,input_shape,num_actions,self.target_net)

                # For Dueling Network:
                elif model_type == 'dueling':
                        self.create_dueling_dqn_model(window,input_shape,num_actions,self.source_net)
                        if target_fix_flag:
                                self.create_dueling_dqn_model(window,input_shape,num_actions,self.target_net)

        def create_linear_model(self, window, input_shape, num_actions, model_name):
                """
                Create Linear network

                Parameters
                ----------
                window: int
                  Each input to the network is a sequence of frames. This value
                  defines how many frames are in the sequence.
                input_shape: tuple(int, int)
                  The expected input image size.
                num_actions: int
                  Number of possible actions. Defined by the gym environment.
                """
                # input placeholders

                with tf.name_scope(model_name) as scope:
                        
                        # Input and target placeholders.
                        self.x = tf.placeholder(tf.float32,shape=[None,input_shape[0],input_shape[1],window],name='input_state')  
                        self.y_true= tf.placeholder(tf.float32,shape=[None,1],name='target_q_val')     

                        # Reshape input. 
                        self.x_flat = tf.reshape(self.x,[-1,84*84*4],name='flat_input')

                        # linear layer
                        self.W = tf.Variable(tf.truncated_normal([84*84*4,num_actions],stddev=0.1),name='weight')
                        self.b = tf.Variable(tf.constant(0.1,shape=[num_actions]),name='bias')

                        # Extract predicted Q values. 
                        self.pred_q = tf.add(tf.matmul(self.x_flat,self.W),self.b,name='pred_q')
                        # Selected Action is a one-hot encoding of which actions were chosen.
                        self.selected_action = tf.placeholder(tf.float32,shape=[None,self.num_actions],name='selected_action')
                                        
                        # Create the following summaries only for the source network. 
                        if model_name.startswith('source'):

                                # Predicted y: Q values for teh action selected. 
                                self.pred_y = tf.reduce_sum(tf.multiply(self.pred_q,self.selected_action),axis=1)

                                # Loss
                                self.loss = mean_huber_loss(self.y_true, self.pred_y)

                                # Evaluated Reward. 
                                self.accumulated_avg_reward = tf.placeholder(tf.float32, shape=(),name='accumulated_avg_reward')

                                # Train variable.
                                self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss,name='Adam_minimizer')                             
                                self.maxq_summary = tf.summary.scalar('Max_Q',tf.reduce_max(self.pred_q))
                                self.loss_summary = tf.summary.scalar('Loss',self.loss)                 

                                self.merged = tf.summary.merge_all()

                                # VARIABLE AND SUMMARY for training reward. 
                                self.train_reward_val = tf.placeholder(tf.float32, shape=(),name='train_reward_val')
                                self.reward_train_summary = tf.summary.scalar('Training_reward',self.train_reward_val)                          

                                # Evaluation reward summary. 
                                self.reward_summary = tf.summary.scalar('Average_reward',self.accumulated_avg_reward)                   
                        
        def create_dqn_model(self, window, input_shape, num_actions, model_name):  # noqa: D103
                """Create the Q-network model.

                We highly recommend that you use tf.name_scope as discussed in
                class when creating the model and the layers. This will make it
                far easier to understnad your network architecture if you are
                logging with tensorboard.

                Parameters
                ----------
                window: int
                  Each input to the network is a sequence of frames. This value
                  defines how many frames are in the sequence.
                input_shape: tuple(int, int)
                  The expected input image size.
                num_actions: int
                  Number of possible actions. Defined by the gym environment.

                """
                # input placeholders

                with tf.name_scope(model_name) as scope:

                        # Input and target placeholders. 
                        self.x = tf.placeholder(tf.float32,shape=[None,input_shape[0],input_shape[1],window],name='input_state')
                        self.y_true = tf.placeholder(tf.float32,shape=[None,1],name='target_q_val')

                        # conv1 layer
                        self.W_conv1 = tf.Variable(tf.truncated_normal([8,8,window,16],stddev=0.1),name='W_conv1')
                        self.b_conv1 = tf.Variable(tf.constant(0.1,shape=[16]),name='b_conv1')

                        self.conv1 = tf.nn.conv2d(self.x,self.W_conv1,strides=[1,4,4,1],padding='VALID') + self.b_conv1
                        self.relu_conv1 = tf.nn.relu(self.conv1)

                        #conv2 layer
                        self.W_conv2 = tf.Variable(tf.truncated_normal([4,4,16,32],stddev=0.1),name='W_conv2')
                        self.b_conv2 = tf.Variable(tf.constant(0.1,shape=[32]),name='b_conv2')

                        self.conv2 = tf.nn.conv2d(self.relu_conv1,self.W_conv2,strides=[1,2,2,1],padding='VALID') + self.b_conv2
                        self.relu_conv2 = tf.nn.relu(self.conv2)
                        self.relu_conv2_flat = tf.reshape(self.relu_conv2, [-1,9*9*32])

                        # fc3 layer
                        self.W_fc3 = tf.Variable(tf.truncated_normal([9*9*32,256], stddev=0.1),name='W_fc3')
                        self.b_fc3 = tf.Variable(tf.constant(0.1,shape=[256]),name='b_fc3')

                        self.fc3 = tf.matmul(self.relu_conv2_flat,self.W_fc3) + self.b_fc3
                        self.relu_fc3 = tf.nn.relu(self.fc3)
                        
                        # output layer
                        self.W_fc4 = tf.Variable(tf.truncated_normal([256,num_actions], stddev=0.1),name='W_output')
                        self.b_fc4 = tf.Variable(tf.constant(0.1,shape=[num_actions]),name='b_output')


                        # Selected Action is a one-hot encoding of which actions were chosen.
                        self.selected_action = tf.placeholder(tf.float32,shape=[None,self.num_actions],name='selected_action')
                        # Extract predicted Q values. 
                        self.pred_q = tf.add(tf.matmul(self.relu_fc3,self.W_fc4),self.b_fc4,name='pred_q')
                        
                        # For the source network, and not target network. 
                        if model_name.startswith('source'):

                                # Predicted Q at the executed action.
                                self.pred_y = tf.reduce_sum(tf.multiply(self.pred_q,self.selected_action),axis=1)
                                # Lloss value 
                                self.loss = mean_huber_loss(self.y_true, self.pred_y)

                                # Evaluation reward.
                                self.accumulated_avg_reward = tf.placeholder(tf.float32, shape=(),name='accumulated_avg_reward')

                                # Train with ADAM.
                                self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss,name='Adam_minimizer')                                             

                                self.maxq_summary = tf.summary.scalar('Max_Q',tf.reduce_max(self.pred_q))
                                self.loss_summary = tf.summary.scalar('Loss',self.loss)
                                        
                                self.merged = tf.summary.merge_all()

                                # VARIABLE AND SUMMARY for training reward. 
                                self.train_reward_val = tf.placeholder(tf.float32, shape=(),name='train_reward_val')
                                self.reward_train_summary = tf.summary.scalar('Training_reward',self.train_reward_val)  

                                self.reward_summary = tf.summary.scalar('Average_reward',self.accumulated_avg_reward)

        def create_dueling_dqn_model(self, window, input_shape, num_actions, model_name):  # noqa: D103
                
                with tf.name_scope(model_name) as scope:

                        self.x = tf.placeholder(tf.float32,shape=[None,input_shape[0],input_shape[1],window],name='input_state')
                        self.y_true = tf.placeholder(tf.float32,shape=[None,1],name='target_q_val')

                        # conv1 layer
                        self.W_conv1 = tf.Variable(tf.truncated_normal([8,8,window,16],stddev=0.1),name='W_conv1')
                        self.b_conv1 = tf.Variable(tf.constant(0.1,shape=[16]),name='b_conv1')

                        self.conv1 = tf.nn.conv2d(self.x,self.W_conv1,strides=[1,4,4,1],padding='VALID') + self.b_conv1
                        self.relu_conv1 = tf.nn.relu(self.conv1)

                        #conv2 layer
                        self.W_conv2 = tf.Variable(tf.truncated_normal([4,4,16,32],stddev=0.1),name='W_conv2')
                        self.b_conv2 = tf.Variable(tf.constant(0.1,shape=[32]),name='b_conv2')

                        self.conv2 = tf.nn.conv2d(self.relu_conv1,self.W_conv2,strides=[1,2,2,1],padding='VALID') + self.b_conv2
                        self.relu_conv2 = tf.nn.relu(self.conv2)
                        self.relu_conv2_flat = tf.reshape(self.relu_conv2, [-1,9*9*32])

        # # # # # # # # # # # # # : Remember, the convolutional layers are shared, but not the fully connected # # # # # # # # # # # # # #

                        # Now splitting into advantage and value streams, using 512 hidden units for Dueling architecture.
                        
                        # Advantage Stream:
                        self.W_fc3_adv = tf.Variable(tf.truncated_normal([9*9*32,512], stddev=0.1),name='W_fc3_adv')
                        self.b_fc3_adv = tf.Variable(tf.constant(0.1,shape=[512]),name='b_fc3_adv')

                        self.fc3_adv = tf.matmul(self.relu_conv2_flat,self.W_fc3_adv) + self.b_fc3_adv
                        self.relu_fc3_adv = tf.nn.relu(self.fc3_adv)
                                        
                        self.W_fc4_adv = tf.Variable(tf.truncated_normal([512,num_actions],stddev=0.1),name='W_fc4_adv')
                        self.b_fc4_adv = tf.Variable(tf.constant(0.1,shape=[num_actions]),name='b_output')

                        self.fc4_adv = tf.matmul(self.relu_fc3_adv,self.W_fc4_adv) + self.b_fc4_adv

                        # Value Stream:
                        self.W_fc3_val = tf.Variable(tf.truncated_normal([9*9*32,512],stddev=0.1),name='W_fc3_val')
                        self.b_fc3_val = tf.Variable(tf.constant(0.1,shape=[512]),name='b_fc3_val')

                        self.fc3_val = tf.matmul(self.relu_conv2_flat,self.W_fc3_val) + self.b_fc3_val
                        self.relu_fc3_val = tf.nn.relu(self.fc3_val)

                        self.W_fc4_val = tf.Variable(tf.truncated_normal([512,1],stddev=0.1),name='W_fc4_val')          
                        self.b_fc4_val = tf.Variable(tf.constant(0.1,shape=[1]),name='b_fc4_val')

                        self.fc4_val = tf.matmul(self.relu_fc3_val,self.W_fc4_val) + self.b_fc4_val

                        # Merging into Q values (subtracting out the average advantage to disambiguate the value and advantage.)
                        self.pred_q = tf.add(self.fc4_val, tf.subtract(self.fc4_adv,tf.reduce_mean(self.fc4_adv)),name='pred_q')
                        self.selected_action = tf.placeholder(tf.float32,shape=[None,self.num_actions],name='selected_action')

                        if model_name.startswith('source'):
                                self.pred_y = tf.reduce_sum(tf.multiply(self.pred_q,self.selected_action),axis=1)
                                self.loss = mean_huber_loss(self.y_true, self.pred_y)
                                self.accumulated_avg_reward = tf.placeholder(tf.float32, shape=(),name='accumulated_avg_reward')
                                self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss,name='Adam_minimizer')                                             
                                self.maxq_summary = tf.summary.scalar('Max_Q',tf.reduce_max(self.pred_q))
                                self.loss_summary = tf.summary.scalar('Loss',self.loss)

                                self.merged = tf.summary.merge_all()

                                self.train_reward_val = tf.placeholder(tf.float32, shape=(),name='train_reward_val')
                                self.reward_train_summary = tf.summary.scalar('Training_reward',self.train_reward_val)                  
                                
                                self.reward_summary = tf.summary.scalar('Average_reward',self.accumulated_avg_reward)

        def calc_q_values(self, state,net_namescope):
                """Given a state (or batch of states) calculate the Q-values.

                Basically run your network on these states.

                Return
                ------
                Q-values for the state(s)
                """
                
                # Specify which tensors to retrieve from forward pass.
                prediction_tensor = tf.get_default_graph().get_tensor_by_name(net_namescope+'/pred_q:0')
                input_tensor = tf.get_default_graph().get_tensor_by_name(net_namescope+'/input_state:0')

                # RUn a forward pass either for a batch or single state.
                if state.shape[0]==32:
                        #q_vals = self.sess.run(self.pred_q,{self.x: state.reshape(32,84,84,4)})
                        # q_vals = self.sess.run(prediction_tensor,{input_tensor: state.reshape(32,84,84,4)})
                        # q_vals = self.sess.run(prediction_tensor,{input_tensor: state.swapaxes(1,2).swapaxes(2,3)})
                        # q_vals = self.sess.run(prediction_tensor,{input_tensor: np.moveaxis(state,1,3)})
                        q_vals = self.sess.run(prediction_tensor,{input_tensor: state})
                else:
                        #q_vals = self.sess.run(self.pred_q,{self.x: state.reshape(1,84,84,4)})
                        state_extra = state.reshape(1,84,84,4)
                        # q_vals = self.sess.run(prediction_tensor,{input_tensor: state.reshape(1,84,84,4)})
                        # q_vals = self.sess.run(prediction_tensor,{input_tensor: state_extra.swapaxes(1,2).swapaxes(2,3)})
                        # q_vals = self.sess.run(prediction_tensor,{input_tensor: np.moveaxis(state_extra,1,3)})                        
                        q_vals = self.sess.run(prediction_tensor,{input_tensor: state_extra})                   

                # print(q_vals)
                return q_vals
        
        def select_action(self, state, is_training, net_namescope, iteration_number=None):

                """Select the action based on the current state.

                You will probably want to vary your behavior here based on
                which stage of training your in. For example, if you're still
                collecting random samples you might want to use a
                UniformRandomPolicy.

                If you're testing, you might want to use a GreedyEpsilonPolicy
                with a low epsilon.

                If you're training, you might want to use the
                LinearDecayGreedyEpsilonPolicy.

                This would also be a good place to call
                process_state_for_network in your preprocessor.

                Returns
                --------
                selected action
                """

                q_vals = self.calc_q_values(state,net_namescope)

                # Use a linear decay greedy epsilon policy to choose actions for training.
                if is_training:
                        action_index = self.policy.select_action(q_vals,is_training,iteration_number)
                if not(is_training):
                        epsilon = 0.05

                        # For evaluating use greedy epsilon.
                        pol = GreedyEpsilonPolicy(epsilon,self.num_actions)
                        action_index = pol.select_action(q_vals)

                return action_index

        def update_policy(self, samples):
                """Update your policy.

                Behavior may differ based on what stage of training your   
                in. If you're in training mode then you should check if you
                should update your network parameters based on the current
                step and the value you set for train_freq.

                Inside, you'll want to sample a minibatch, calculate the
                target values, update your network, and then update your
                target values.

                You might want to return the loss and other metrics as an
                output. They can help you monitor how training is going.
                """
                # self.sess = tf.get_default_session()

                # Process a batch of states for the network.
                samples_for_train = self.preprocessor.process_batch(samples)
        
                batch_state = np.zeros((32,84,84,4))
                batch_next_state = np.zeros((32,84,84,4))

                batch_actions = np.zeros((32,self.num_actions))

                # Creating numpy variables to feed to TensorFlow.
                for i in range(len(samples_for_train)):
                        batch_state[i] = samples_for_train[i].state
                        batch_next_state[i] = samples_for_train[i].next_state
                        batch_actions[i, samples_for_train[i].action] = 1.
                targets = np.zeros((32,1))

                # Based on the model type, run a forward pass.
                # Calculate the target and the predicted Q values.
                if self.model_type == 'linear' or self.model_type == 'dqn':
                        # use source network
                        q_state = self.calc_q_values(batch_state,self.source_net)
                        if self.target_fix_flag:
                                # use target network
                                q_next_state = self.calc_q_values(batch_next_state,self.target_net)
                        else:
                                # use source network
                                q_next_state = self.calc_q_values(batch_next_state,self.source_net)

                        for i in range(len(targets)):
                                targets[i] = samples_for_train[i].reward + self.gamma*np.max(q_next_state[i])

                # For double networks, use the target network to compute teh target values.
                elif self.model_type == 'double_linear' or self.model_type == 'double_dqn' or self.model_type == 'dueling':
                # use source network
                        q_state = self.calc_q_values(batch_state,self.source_net)
                        # use source network to find argmax action
                        max_action = np.argmax(self.calc_q_values(batch_next_state,self.source_net),axis=1)
                        # use target network to find target q value
                        q_next_state = self.calc_q_values(batch_next_state,self.target_net)

                        for i in range(len(targets)):
                                targets[i] = samples_for_train[i].reward + self.gamma*q_next_state[i,max_action[i]]

                # Specify tensors to call from TensorFlow run.
                input_state_tensor = tf.get_default_graph().get_tensor_by_name(self.source_net + '/input_state:0')
                input_target_tensor = tf.get_default_graph().get_tensor_by_name(self.source_net + '/target_q_val:0')
                selected_action_tensor = tf.get_default_graph().get_tensor_by_name(self.source_net + '/selected_action:0')
                
                # Run a forward pass and extract the loss, predicted y values, and summary.
                merged_summary, loss_value, pred_y, _ = self.sess.run([self.merged, self.loss, self.pred_y, self.train],feed_dict={input_state_tensor: batch_state, input_target_tensor: targets, selected_action_tensor: batch_actions, self.accumulated_avg_reward: self.average_reward})                             
                
                # To monitor:
                #print("Loss Value:", loss_value)
                #print("Predicted Q Value:", pred_y)
                #print("Target Q Value:", target_y)

                return merged_summary, loss_value               

        def fit(self, env, num_iterations, max_episode_length=None):
                """Fit your model to the provided environment.

                Its a good idea to print out things like loss, average reward,
                Q-values, etc to see if your agent is actually improving.

                You should probably also periodically save your network
                weights and any other useful info.

                This is where you should sample actions from your network,
                collect experience samples and add them to your replay memory,
                and update your network parameters.

                Parameters
                ----------
                env: gym.Env
                  This is your Atari environment. You should wrap the
                  environment using the wrap_atari_env function in the
                  utils.py
                num_iterations: int
                  How many samples/updates to perform.
                max_episode_length: int
                  How long a single episode should last before the agent
                  resets. Can help exploration.
                """
                
                # INITIALIZE THE FILE WRITING AND SUMMARIES.
                train_folder = 'train_' + self.model_type + '/'
                if not os.path.exists(train_folder):
                        os.makedirs(train_folder)
                train_writer = tf.summary.FileWriter(train_folder, self.sess.graph)

                # If starting from scratch:
                if not self.is_restored:
                        init = tf.global_variables_initializer()
                        self.sess.run(init)

                saver = tf.train.Saver(max_to_keep=None)

                # First we burn into the memory.
                counter = 0
                eps_counter = 0 

                # First reset the environment.
                state = self.preprocessor.process_state_for_memory(env.reset(),size=1)
                next_state = np.zeros((84,84))  

                # While we haven't burned in completely.
                while counter<self.num_burn_in:

                        # Create a sample instance to store in the memory. 
                        sample = Single_Sample()
                        sample.state = state
                        sample.action = np.random.randint(0,self.num_actions)

                        # Run the environment forward.
                        [state_temp_var, sample.reward, sample.is_terminal, dummy] = env.step(sample.action)            

                        # Process the reward and state for memory.
                        sample.next_state = self.preprocessor.process_state_for_memory(state_temp_var, size=1)
                        sample.reward = self.preprocessor.process_reward(sample.reward)

                        state = sample.next_state
                        counter += 1
                        eps_counter += 1

                        # If we have reached maximum episode length or if the episode terminated, reset he environment.
                        if (eps_counter>=max_episode_length-1)or(sample.is_terminal):
                                sample.is_terminal = 1
                                state = self.preprocessor.process_state_for_memory(env.reset(),size=1)
                                eps_counter = 0
                        
                        # Write sample to memory
                        self.memory.append(sample) 

                print('Burn in completed')
                
                # Take 4 random steps to fill up the state instead of 0 padding.
                state = np.zeros((84,84,4))
                state[:,:,0] = self.preprocessor.process_state_for_memory(env.reset(),size=1)
                state[:,:,1] = self.preprocessor.process_state_for_memory(env.step(0)[0],size=1) 
                state[:,:,2] = self.preprocessor.process_state_for_memory(env.step(0)[0],size=1) 
                state[:,:,3] = self.preprocessor.process_state_for_memory(env.step(0)[0],size=1) 

                # Now for the maximum number of iterations specified, loop:
                num_iter = 0
                counter = 0
                eps_counter = 0

                while num_iter <= num_iterations:

                        # If we have reached 1/3 or 2/3, then evaluate the model.
                        if num_iter == 0 or num_iter == num_iterations/3 or num_iter == (2*num_iterations)/3 or num_iter == num_iterations:
                                
                                # Run evaluation of the model.
                                #self.average_reward = self.evaluate(env,20,10*max_episode_length,num_iter,lambda x: x%5==0,True)
                                self.average_reward = self.evaluate(env,20,max_episode_length,num_iter,False,True)

                                # Retrieve evaluated rewards.
                                summary_reward, _ = self.sess.run([self.reward_summary,self.accumulated_avg_reward],{self.accumulated_avg_reward:self.average_reward})
                                train_writer.add_summary(summary_reward, num_iter)
                        
                        # Create a single sample element to retrieve another step from the environment.
                        sample = Single_Sample()
                        # Select teh action to take based on the computed Q values.
                        sample.action = self.select_action(state,1,self.source_net,iteration_number=num_iter)
                        sample.state = copy.deepcopy(state[:,:,3])

                        # Roll the states back to make way for a new frame.
                        state[:,:,0:3] = state[:,:,1:4]
                        [state_temp_var,sample.reward,sample.is_terminal,dummy] = env.step(sample.action)

                        # COLLECTING TRAINING REWARD SUMMARY:
                        # print("Reached Training Reward Summary")
                        summary_train_reward = self.sess.run(self.reward_train_summary,{self.train_reward_val: sample.reward})
                        # print("Value of Training Reward:", sample.reward)
                        train_writer.add_summary(summary_train_reward, num_iter)

                        # Process the reward and append the sample to memory.
                        sample.reward = self.preprocessor.process_reward(sample.reward)
                        self.memory.append(sample)

                        # Reset the environment if the episode reaches maximum length, or if terminated.
                        if (eps_counter>=max_episode_length-1)or(sample.is_terminal):
                                sample.is_terminal = 1
                                state[:,:,3] = self.preprocessor.process_state_for_memory(env.reset(),size=1)
                                eps_counter = 0
                        else:
                                state[:,:,3] = self.preprocessor.process_state_for_memory(state_temp_var, size=1)

                        if self.replay_memory_flag:
                                # TO RUN WITH EXPERIENCE REPLAY, sample a batch accordingly.
                                #print("Running with experience replay.")
                                batch_samples_obj = self.memory.sample(32)

                                # batch_samples_obj = cProfile.runctx(self.memory.sample(32),globals(),locals())
                        else:
                                # TO RUN WITHOUT EXPERIENCE REPLAY:
                                #print("Running without experience replay.")
                                batch_samples_obj = self.memory.sample_without_replay(32)

                        # Process the data for network.
                        for i in range(32):
                                batch_samples_obj[i].state = self.preprocessor.process_state_for_network(batch_samples_obj[i].state)                            
                                batch_samples_obj[i].next_state = self.preprocessor.process_state_for_network(batch_samples_obj[i].next_state)                          

                        # Call Update policy to train.
                        merged_summary, loss = self.update_policy(batch_samples_obj)
                        
                        # Write summaries to file.
                        train_writer.add_summary(merged_summary, num_iter)

                        num_iter += 1
                        eps_counter += 1

                        # Print.
                        if num_iter % 2000 == 0:
                                print('Iter: ' + str(num_iter), 'Loss: ' + str(loss))

                        # Update the target.
                        if (num_iter % self.target_update_freq == 0) and self.target_fix_flag:
                                self.sess.run(get_hard_target_model_updates(self.target_net, self.source_net))
                                print('Target network updated')

                        # Save the model.
                        if (num_iter % self.save_every) == 0:
                                save_folder = 'saved_models_' + self.model_type
                                if not os.path.exists:
                                        os.makedirs(save_folder)
                                save_path = saver.save(self.sess, save_folder + '/model_' + str(num_iter) + '.ckpt')
                                print('Model saved in file: %s' % save_path)

                                # evaluate it again.
                                self.average_reward = self.evaluate(env,20,10*max_episode_length,num_iter,False,False)
                                summary_reward,_ = self.sess.run([self.reward_summary,self.accumulated_avg_reward],{self.accumulated_avg_reward:self.average_reward})
                                train_writer.add_summary(summary_reward, num_iter)

                train_writer.close()

        def evaluate(self, env, num_episodes, max_episode_length, monitor_count, vc_func, f, render=False):
                """Test your agent with a provided environment.
                
                You shouldn't update your network parameters here. Also if you
                have any layers that vary in behavior between train/test time
                (such as dropout or batch norm), you should set them to test.

                Basically run your policy on the environment and collect stats
                like cumulative reward, average episode length, etc.

                You can also call the render function here if you want to
                visually inspect your policy.
                """
                state = np.zeros((84,84,4))
                average_reward = 0.0
                folder_name = './monitor_' + self.model_type
                if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                env_mon = wrappers.Monitor(env, folder_name + '/' + env.spec._env_name + '-experiment-' + str(monitor_count), video_callable=vc_func, mode='evaluation', force=f)
                for i_episode in range(num_episodes):
                        state[:,:,0] = self.preprocessor.process_state_for_memory(env_mon.reset(),size=1)
                        state[:,:,1] = self.preprocessor.process_state_for_memory(env_mon.step(0)[0],size=1) 
                        state[:,:,2] = self.preprocessor.process_state_for_memory(env_mon.step(0)[0],size=1) 
                        state[:,:,3] = self.preprocessor.process_state_for_memory(env_mon.step(0)[0],size=1)
                        for t in range(max_episode_length):
                                if render:
                                        env_mon.render()
                                # select action here.
                                action = self.select_action(state,0,self.source_net)
                                observation, reward, done, info = env_mon.step(action)
                                state[:,:,0:3] = state[:,:,1:4]
                                state[:,:,3] = self.preprocessor.process_state_for_memory(observation, size=1)
                                average_reward += reward
                                if done:
                                        print("Episode finished after {} timesteps".format(t+1))
                                        break
                average_reward /= num_episodes
                return average_reward

        def restore_model(self, model_file):
                saver = tf.train.Saver()
                saver.restore(self.sess, model_file)
                self.is_restored = True

