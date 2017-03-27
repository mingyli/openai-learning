'''
initialize replay memory D
initialize action-value function Q with random weights
observe initial state s
for each episode:
	select an action a
		use epsilon greedy exploration
	carry out action a
	observe reward r and new state s'
	store experience <s, a, r, s'> in replay memory D

	sample random transitions <ss, aa, rr, ss'> from replay memory D
	calculate target for each minibatch transition:
		if ss' is terminal state then tt = rr
		else tt = rr + gamma * max_action Q(ss', aa')
	train Q network using loss (tt - Q(ss, aa))
	s = s'



Deep Deterministic Policy Gradient
==================================
for use with continuous action

initialize critic network Q(s,a) and actor mu(s) 
initialize target network Q' and mu' 
initialize replay buffer R
for each episode:
	initialize random process N for action exploration # DDPG doesn't use epsilon-greedy? N is noise?
	observe initial state s_1
	for t = 1 -> T:
		select action a_t = mu(s_t) + N_t
		carry out action a_t
		observe reward r_t and new state s_t+1
		store transition <s_t, a_t, r_t, s_t+1> in R

		sample random minibatch of transitions from R
		set y_i = r_i + discount * Q'(s_i+1, mu'(s_i+1))
		update critic Q by minimizing loss L
		update actor network
		update target networks
'''

