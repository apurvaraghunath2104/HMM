import argparse

class HMM:
	"""
		Arguments:

		states: Sequence of strings representing all states
		vocab: Sequence of strings representing all unique observations
		trans_prob: Transition probability matrix. Each cell (i, j) contains P(states[j] | states[i])
		obs_likelihood: Observation likeliood matrix. Each cell (i, j) contains P(vocab[j] | states[i])
		initial_probs: Vector representing initial probability distribution. Each cell i contains P(states[i] | START)
	"""
	def __init__(self, states, vocab, trans_prob, obs_likelihood, initial_probs):
		self.states = states
		self.vocab = vocab
		self.trans_prob = trans_prob
		self.obs_likelihood = obs_likelihood
		self.initial_probs = initial_probs

    # Function to return transition probabilities P(q1|q2)
	def tprob(self, q1, q2):
		if not (q1 in self.states and q2 in ['START'] + self.states):
			raise ValueError("invalid input state(s)")
		q1_idx = self.states.index(q1)
		if q2 == 'START':
			return self.initial_probs[q1_idx]
		q2_idx = self.states.index(q2)
		return self.trans_prob[q2_idx][q1_idx]

    # Function to return observation likelihood P(o|q)
	def oprob(self, o, q):
		if not o in self.vocab:
			raise ValueError('invalid observation')
		if not (q in self.states and q != 'START'):
			raise ValueError('invalid state')
		obs_idx = self.vocab.index(o)
		state_idx = self.states.index(q)
		return self.obs_likelihood[obs_idx][state_idx]

	# Function to retrieve all states
	def get_states(self):
		return self.states.copy()



def initialize_icecream_hmm():
	states = ['HOT', 'COLD']
	vocab = ['1', '2', '3']
	tprob_mat = [[0.7, 0.3], [0.4, 0.6]]
	obs_likelihood = [[0.2, 0.5], [0.4, 0.4], [0.4, 0.1]]
	initial_prob = [0.8, 0.2]
	hmm = HMM(states, vocab, tprob_mat, obs_likelihood, initial_prob)
	return hmm


# Function to implement viterbi algorithm
# Arguments:
# hmm: An instance of HMM class as defined in this file
# obs: A string of observations, e.g. ("132311")
# Returns: seq, prob
# Where, seq (list) is a list of states showing the most likely path and prob (float) is the probability of that path
def viterbi(hmm, obs):
	initial_prob_list =[]
	obs_likelihood_list = []
	tprob_mat_list = []
	res_list = []
	path_list = []
	temp =[]
	for i in hmm.initial_probs:
		initial_prob_list.append(i)
	for i in hmm.trans_prob:
		tprob_mat_list.append(i)
	initial_prob_hot = initial_prob_list[0] *  hmm.oprob(obs[0],'HOT')
	initial_prob_cold = initial_prob_list[1] *  hmm.oprob(obs[0],'COLD')
	res_list.append(initial_prob_hot)
	res_list.append(initial_prob_cold)
	temp = []
	temp.append(res_list)
	for i in range(1,len(obs)):
		backtrack = []
		res_list = []
		total1 = tprob_mat_list[0][0] * hmm.oprob(obs[i],'HOT') * temp[i-1][0]
		total2 = tprob_mat_list[1][0] * hmm.oprob(obs[i],'HOT') * temp[i-1][1]
		max1 = max(total1, total2)
		res_list.append(max1)
		if total1 > total2:
			backtrack.append('HOT')
		else:
			backtrack.append('COLD')

		sum1 = tprob_mat_list[0][1] * hmm.oprob(obs[i],'COLD') * temp[i-1][0]
		sum2 = tprob_mat_list[1][1] * hmm.oprob(obs[i],'COLD') * temp[i-1][1]
		max2 = max(sum1, sum2)
		res_list.append(max2)
		if sum1 > sum2:
			backtrack.append('HOT')
		else:
			backtrack.append('COLD')
		temp.append(res_list)
		path_list.append(backtrack)
	final_backpointer = []
	if temp[-1][0] > temp[-1][1]:
		day = 'HOT'
		final_backpointer.append('HOT')
	elif temp[-1][0] == temp[-1][1]:
		day = hmm.states[0]
		final_backpointer.append(hmm.states[0])
	else:
		day = 'COLD'
		final_backpointer.append('COLD')
	for i in range(len(path_list)-1, -1,-1):
		if temp[i][0] == temp[i][1]:
			final_backpointer.append(hmm.states[0])
			day = hmm.states[0]
		else:
			if day == 'HOT':
				final_backpointer.append(path_list[i][0])
				day = path_list[i][0]
			else:
				final_backpointer.append(path_list[i][1])
				day = path_list[i][1]
	final_backpointer = final_backpointer[::-1]
	final_max_prob = max(res_list)

	return final_backpointer, final_max_prob



def main():
	# We can initialize our HMM using initialize_icecream_hmm function
	hmm = initialize_icecream_hmm()

	# We can retrieve all states as
	print("States: {0}".format(hmm.get_states()))

	# We can get transition probability P(HOT|COLD) as
	prob = hmm.tprob('HOT', 'COLD')
	print("P(HOT|COLD) = {0}".format(prob))

	# We can get transition probability P(COLD|START) as
	prob = hmm.tprob('COLD', 'START')
	print("P(COLD|START) = {0}".format(prob))

	# We can get observation likelihood P(1|COLD) as
	prob = hmm.oprob('1', 'COLD')
	print("P(1|COLD) = {0}".format(prob))

	# We can get observation likelihood P(2|HOT) as
	prob = hmm.oprob('2', 'HOT')
	print("P(2|HOT) = {0}".format(prob))

	# You should call the viterbi algorithm as
	path, prob = viterbi(hmm, "111")
	print("Path: {0}".format(path))
	print("Probability: {0}".format(prob))


################ Do not make any changes below this line ################
if __name__ == '__main__':
    exit(main())
