"""Core classes."""

import numpy as np
from PIL import Image
import copy

class Sample:

	"""Represents a reinforcement learning sample.

	Used to store observed experience from an MDP. Represents a
	standard `(s, a, r, s', terminal)` tuple.

	Note: This is not the most efficient way to store things in the
	replay memory, but it is a convenient class to work with when
	sampling batches, or saving and loading samples while debugging.

	Parameters
	----------
	state: array-like
	  Represents the state of the MDP before taking an action. In most
	  cases this will be a numpy array.
	action: int, float, tuple
	  For discrete action domains this will be an integer. For
	  continuous action domains this will be a floating point
	  number. For a parameterized action MDP this will be a tuple
	  containing the action and its associated parameters.
	reward: float
	  The reward received for executing the given action in the given
	  state and transitioning to the resulting state.
	next_state: array-like
	  This is the state the agent transitions to after executing the
	  `action` in `state`. Expected to be the same type/dimensio	git ns as
	  the state.
	is_terminal: boolean
	  True if this action finished the episode. False otherwise.
	"""
	def __init__(self):
	  
		# THIS CLASS STORES an <S,A,R,S',(is_terminal)> tuple, where the state consists of 4 frames. 
		self.state = np.zeros((84,84,4))

		self.action = np.zeros(1,dtype=int)
		self.reward = np.zeros(1,dtype=float)
		self.is_terminal = np.zeros(1,dtype=bool)
		
		# self.next_state = np.zeros((4,84,84))
		self.next_state = np.zeros((84,84,4))

class Single_Sample:

	def __init__(self):

		# For efficient memory usage, we store single frames and assemble them 
		# before training to states.
	  
		self.state = np.zeros((84,84))
		self.action = np.zeros(1,dtype=int)
		self.reward = np.zeros(1,dtype=float)
		self.is_terminal = np.zeros(1,dtype=bool)

class Preprocessor:
	"""Preprocessor base class.

	This is a suggested interface for the preprocessing steps. You may
	implement any of these functions. Feel free to add or change the
	interface to suit your needs.

	Preprocessor can be used to perform some fixed operations on the
	raw state from an environment. For example, in ConvNet based
	networks which use image as the raw state, it is often useful to
	convert the image to greyscale or downsample the image.

	Preprocessors are implemented as class so that they can have
	internal state. This can be useful for things like the
	AtariPreproccessor which maxes over k frames.

	If you're using internal states, such as for keeping a sequence of
	inputs like in Atari, you should probably call reset when a new
	episode begins so that state doesn't leak in from episode to
	episode.
	"""
	def __init__(self):
		pass

	def process_state_for_network(self, state):
		"""Preprocess the given state before giving it to the network.

		Should be called just before the action is selected.

		This is a different method from the process_state_for_memory
		because the replay memory may require a different storage
		format to reduce memory usage. For example, storing images as
		uint8 in memory is a lot more efficient thant float32, but the
		networks work better with floating point images.

		Parameters
		----------
		state: np.ndarray
		  Generally a numpy array. A single state from an environment.

		Returns
		-------
		processed_state: np.ndarray
		  Generally a numpy array. The state after processing. Can be
		  modified in anyway.

		"""
		# Normalization Schemes.
		# state = state.astype(np.float32)

		# for i in range(4):
			# state[i] -= np.min(state[i])
			# state[i] /= np.max(state[i])
		#state /= 255.
			# print(state[i].shape, np.min(state[i]).shape)

		# Converts the memory states to float for training. 
		return state.astype(np.float32)

	def process_state_for_memory(self, state, size=4):
		"""Preprocess the given state before giving it to the replay memory.

		Should be called just before appending this to the replay memory.

		This is a different method from the process_state_for_network
		because the replay memory may require a different storage
		format to reduce memory usage. For example, storing images as
		uint8 in memory and the network expecting images in floating
		point.

		Parameters
		----------
		state: np.ndarray
		  A single state from an environmnet. Generally a numpy array.

		Returns
		-------
		processed_state: np.ndarray
		  Generally a numpy array. The state after processing. Can be
		  modified in any manner.

		"""
		k = size   	
		x = np.zeros((84,84,size))
		
		# For all frames, convert ot grayscale and resize.  
		for i in range(k):

			temp = Image.fromarray(state[:,:,i])
			temp = temp.convert('L') 
			# We are directly resizing, not cropping and then resizing. 
			temp = temp.resize((84,84))
			x[:,:,i] = np.array(temp)
		
		if size==1:
			x = x.reshape(84,84)
		return x

	def process_batch(self, samples):
		"""Process batch of samples.

		If your replay memory storage format is different than your
		network input, you may want to apply this function to your
		sampled batch before running it through your update function.

		Parameters
		----------
		samples: list(tensorflow_rl.core.Sample)
		  List of samples to process

		Returns
		-------
		processed_samples: list(tensorflow_rl.core.Sample)
		  Samples after processing. Can be modified in anyways, but
		  the list length will generally stay the same.
		"""

		# For all samples in the batch, process the states for the network. 
		# REmember, rewards were already processed in the fit function.
		processed_sample = [Sample() for i in range(len(samples))]
		# Assuming the samples is a list of objects of Sample Class.
		for i in range(len(samples)):

			processed_sample[i].state = self.process_state_for_network(samples[i].state)
			processed_sample[i].next_state = self.process_state_for_network(samples[i].next_state)
			processed_sample[i].action = samples[i].action
			processed_sample[i].reward = samples[i].reward
			processed_sample[i].is_terminal = samples[i].is_terminal

		# pass
		return processed_sample

		# return samples

	def process_reward(self, reward):
		"""Process the reward.

		Useful for things like reward clipping. The Atari environments
		from DQN paper do this. Instead of taking real score, they
		take the sign of the delta of the score.

		Parameters
		----------
		reward: float
		  Reward to process

		Returns
		-------
		processed_reward: float
		  The processed reward
		"""

		# Reward Clipping.
		if reward>0:
			reward=1
		elif reward<0:
			reward=-1
		else:
			reward=0
		
		return reward

	def reset(self):
		"""Reset any internal state.

		Will be called at the start of every new episode. Makes it
		possible to do history snapshots.
		"""
		pass


class ReplayMemory:
	"""Interface for replay memories.

	We have found this to be a useful interface for the replay
	memory. Feel free to add, modify or delete methods/attributes to
	this class.

	It is expected that the replay memory has implemented the
	__iter__, __getitem__, and __len__ methods.

	If you are storing raw Sample objects in your memory, then you may
	not need the end_episode method, and you may want to tweak the
	append method. This will make the sample method easy to implement
	(just ranomly draw saamples saved in your memory).

	However, the above approach will waste a lot of memory (as states
	will be stored multiple times in s as next state and then s' as
	state, etc.). Depending on your machine resources you may want to
	implement a version that stores samples in a more memory efficient
	manner.

	Methods
	-------
	append(state, action, reward, debug_info=None)
	  Add a sample to the replay memory. The sample can be any python
	  object, but it is suggested that tensorflow_rl.core.Sample be
	  used.
	end_episode(final_state, is_terminal, debug_info=None)
	  Set the final state of an episode and mark whether it was a true
	  terminal state (i.e. the env returned is_terminal=True), of it
	  is is an artificial terminal state (i.e. agent quit the episode
	  early, but agent could have kept running episode).
	sample(batch_size, indexes=None)
	  Return list of samples from the memory. Each class will
	  implement a different method of choosing the
	  samples. Optionally, specify the sample indexes manually.
	clear()
	  Reset the memory. Deletes all references to the samples.
	"""
	# def __init__(self, max_size, window_length):
	def __init__(self, max_size):		
		"""Setup memory.

		You should specify the maximum size o the memory. Once the
		memory fills up oldest values should be removed. You can try
		the collections.deque class as the underlying storage, but
		your sample method will be very slow.

		We recommend using a list as a ring buffer. Just track the
		index where the next sample should be inserted in the list.
		"""

		# Initialize the memory prior to usage, to prevent dynamic allocation. 
		# Python's amortized implementation seems to be slightly slower than preallocation. 
		self.max_size = max_size
		self.memory = [Single_Sample() for i in range(self.max_size)]		

		self.mem_length = 0

		# Index of oldest element in the history.
		self.low_ind = 0

	# def append_old(self, sample):
	# 	# raise NotImplementedError('This method should be overridden')

	# 	if len(self.memory)<self.max_size:
	# 		self.memory.append(sample)
	# 	else:
	# 		self.memory.pop(0)
	# 		self.memory.append(sample)

	def append(self,sample):

		# Until the memory is filled, keep incrementing the "index of length of memory."
		if self.mem_length<self.max_size-1:
			self.memory[self.mem_length]=sample
			self.mem_length += 1

			# print(self.mem_length)

		# Once the memory is filled, start rewriting the oldest history elements by keeping track of low_ind. 
		# Increment low_ind after rewriting. 

		# REMEMBER: The memory is implemented as a circular queue. 
		elif self.mem_length==self.max_size-1:
			self.memory[self.low_ind] = sample
			self.low_ind = (self.low_ind+1)%(self.mem_length+1)


	# def end_episode(self, final_state, is_terminal):
	# 	raise NotImplementedError('This method should be overridden')

	def sample_without_replay(self, batch_size, indexes=None):
		
		# Without Experience Replay, we simply sample in the same order as the samples are embedded in the memory. 
		# Ideally, we do not need to maintain a memory much larger than the batch_size. However we do so here to enable
		# code re-use between sampling with and without replay. 

		# Create a variable of Sample() class, with <SARS> tuples.
		samples = [Sample() for i in range(batch_size)]

		counter = 0
		# Start sampling from the lowest index + 3 (to be able to select 3 previous frames as well).
		dummy_count = (3 + self.low_ind)%self.mem_length

		# Until we have 32 samples:
		while counter<batch_size:

			# SAmpling index incremented. 
			sample_index = dummy_count
			prev_dc = copy.deepcopy(dummy_count)

			# Check that none of the states in the sampled frame or the previous 3 frames have terminal states.
			# This is to ensure we do not sample frames across episodes. 
			while (self.memory[sample_index].is_terminal)or(self.memory[sample_index-1].is_terminal)or(self.memory[sample_index-2].is_terminal)or(self.memory[sample_index-3].is_terminal):					
				dummy_count = (dummy_count+1)%self.mem_length
				sample_index = dummy_count

			# Assemble the batch states from the memory.
			for j in range(4):
				samples[counter].state[:,:,3-j] = self.memory[sample_index-j].state
				samples[counter].next_state[:,:,3-j] = self.memory[(sample_index+1-j)%self.mem_length].state

			# Copy action selected and the reward obtained.
			samples[counter].action = self.memory[sample_index].action
			samples[counter].reward = self.memory[sample_index].reward
			
			counter +=1 

			if (prev_dc==dummy_count):	
				dummy_count = (dummy_count+1)%self.mem_length

		return samples

	def sample(self, batch_size, indexes=None):
		

		# Sampling with replay. THis routine differs from the sampling wihtout replay in that we choose random indices from the memory.
		samples = [Sample() for i in range(batch_size)]
		counter = 0

		# First sample batch_size number of elements.
		sample_indices = np.random.randint(3,high=self.mem_length,size=(batch_size))

		# To check that we have batch_size number of valid samples.
		while counter<batch_size:

			# Select an index.
			sample_index = sample_indices[counter]
			
			# Again check that the smaples are valid, and do not occur across episodes (or across teh boundary of the memory)
			while (self.memory[sample_index].is_terminal)or(self.memory[sample_index-1].is_terminal)or(self.memory[sample_index-2].is_terminal)or(self.memory[sample_index-3].is_terminal)or(abs(sample_index-self.low_ind)<3):				
				# Resample until we obtain a valid sample.
				sample_index = np.random.randint(3,high=self.mem_length)

			# Assemble the state.
			for j in range(4):
				samples[counter].state[:,:,3-j] = self.memory[sample_index-j].state
				samples[counter].next_state[:,:,3-j] = self.memory[sample_index+1-j].state
		
			# Copy over actions and reward.
			samples[counter].action = self.memory[sample_index].action
			samples[counter].reward = self.memory[sample_index].reward
			
			counter +=1 

		return samples

	def clear(self):
		# raise NotImplementedError('This method should be overridden')
		self.memory = [] 
