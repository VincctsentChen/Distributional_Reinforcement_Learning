# =========================================
# Algorithms to find and evaluate policies
# =========================================

# Loading modules
import numpy as np
from gurobipy import *  # Note: copy and paste module files to appropiate interpreter's site-packages folder
from s07_policy_evaluation import evaluate_pi, evaluate_events # policy evaluation

# Function to solve an infinite horizon MDP using the dual formulation of an LP
def lp_mdp_dual(P, r, rterm, alpha, gamma, infeasible, event_states):
    # Infinite horizon MDP using linear programming

    """
    Inputs:
    P is an S x S x T x A array of transition probabilities
    r is an S x T x A array of rewards
    rterm is an S array of terminal rewards 
    alpha is an S x T array of initial transition probabilities
    gamma is the discount factor
    infeasible is a list of nested lists of the index of "clinically infeasible" actions per state and decision epoch
    """""

    """
    Outputs:
        d is the decision rule
        v is the value
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Creating lists of states and actions
    states = list(np.arange(0, S, 1))
    dec_epochs = list(np.arange(0, T, 1))
    epochs = list(np.arange(0, T+1, 1))
    actions = list(np.arange(0, A, 1))

    # Generating list of tuples of state-action pairs
    state_action_pairs = []
    for s in states:
        for t in dec_epochs:
            for a in actions:
                state_action_pairs.append((s, t, a))

    # Creating Gurobi model object
    m = Model()

    # Adding decision variables to model
    y = m.addVars(state_action_pairs)
    yterm = m.addVars(states)

    # Declaring model objective
    m.setObjective(quicksum(r[s, t, a]*y[s, t, a] for s in states for t in dec_epochs for a in actions) +
                   quicksum(rterm[s]*yterm[s] for s in states), GRB.MAXIMIZE)

    # Adding constraints
    const1 = m.addConstrs((quicksum(y[s, 0, a] for a in actions) == alpha[s, 0] for s in states))
    const2 = m.addConstrs((quicksum(y[s, t, a] for a in actions)) -
                          gamma*quicksum(P[ss, s, t-1, aa]*y[ss, t-1, aa]
                                         for ss in states for aa in actions) == alpha[s, t]
                          for s in states for t in dec_epochs[1:])
    const3 = m.addConstrs((yterm[s]-gamma*quicksum(P[ss, s, T-1, aa]*y[ss, T-1, aa]
                                                   for ss in states for aa in actions) == alpha[s, T]
                           for s in states))

    ## Constraint to ensure the policy is feasible
    m.addConstrs((quicksum(y[s, t, ia] for ia in infeasible[s][t]) == 0 for s in states for t in dec_epochs))

    # Processing model specifications
    m.update()

    # Surpressing output
    m.setParam('OutputFlag', False)

    # Setting time limit to 1 hour
    m.setParam('TimeLimit', 3600)

    # Optimizing model
    m.optimize()

    # Storing optimal value of objective function
    J_opt = m.objVal

    # Extracting occupancy measures and decision rule
    d = np.empty((S, T)); d[:] = np.nan
    occup = np.empty((S, T+1, A)); occup[:] = np.nan
    for t in epochs:
        for s in states:
            if t < max(epochs):
                for a in actions:
                    occup[s, t, a] = y[s, t, a].X
                d[s, t] = np.argmax(occup[s, t, :])
            else:
                occup[s, t, 0] = yterm[s].X
                occup[s, t, 1:] = 0

    # Extracting value functions (from primal LP)
    v = np.empty((S, T+1)); v[:] = np.nan
    for t in epochs:
        for s in states:
            if t == 0:
                v[s, t] = const1[s].Pi
            elif t == max(epochs):
                v[s, t] = const3[s].Pi
            else:
                v[s, t] = const2[(s, t)].Pi

    # Calculating expected number of events following policy
    events = evaluate_events(d, P, event_states)

    return v, d, occup, J_opt, events

# Function to solve an infinite horizon MDP with monotonic constraints on the states and decision epochs using the dual formulation of an MIP
def mip_mdp_dual_epochs(P, r, rterm, alpha, gamma, infeasible, event_states, J_opt, warm):
    """
    Inputs:
    P is an S x S x T x A array of transition probabilities
    r is an S x T x A array of rewards
    rterm is an S array of terminal rewards 
    alpha is an S x T array of initial transition probabilities
    gamma is the discount factor
    infeasible is a list of nested lists of the index of "clinically infeasible" actions per state and decision epoch
    """""

    """
    Outputs:
        d is the decision rule
        v is the value
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Creating lists of states and actions
    states = list(np.arange(0, S, 1))
    dec_epochs = list(np.arange(0, T, 1))
    epochs = list(np.arange(0, T+1, 1))
    actions = list(np.arange(0, A, 1))

    # Generating list of tuples of state-action pairs
    state_action_pairs = []
    for s in states:
        for t in dec_epochs:
            for a in actions:
                state_action_pairs.append((s, t, a))

    # Creating Gurobi model object
    m = Model()

    # Adding decision variables to model
    y = m.addVars(state_action_pairs)
    x = m.addVars(state_action_pairs, vtype=GRB.BINARY)
    xterm = m.addVars(states)
    piterm = m.addVars(states, vtype=GRB.BINARY)

    # Warm starting the MIP
    for s in states:
        for t in dec_epochs:
            for a in actions:
                x[s, t, a].start = warm[s, t, a]

    # Declaring model objective
    m.setObjective(quicksum(r[s, t, a]*y[s, t, a] for s in states for t in dec_epochs for a in actions) +
                   quicksum(rterm[s]*xterm[s] for s in states), GRB.MAXIMIZE)

    # Adding constraints
    m.addConstrs((quicksum(y[s, 0, a] for a in actions) == alpha[s, 0] for s in states))
    m.addConstrs((quicksum(y[s, t, a] for a in actions))-gamma*quicksum(P[ss, s, t-1, aa]*y[ss, t-1, aa]
                                                                        for ss in states for aa in actions) == alpha[s, t]
                 for s in states for t in dec_epochs[1:])
    m.addConstrs((xterm[s]-gamma*quicksum(P[ss, s, T-1, aa]*y[ss, T-1, aa]
                                          for ss in states for aa in actions) == alpha[s, T]
                  for s in states))
    m.addConstrs((quicksum(x[s, t, a] for a in actions) == 1 for s in states for t in dec_epochs))
    m.addConstrs((y[s, t, a] <= x[s, t, a] for s in states for t in dec_epochs for a in actions))
    m.addConstrs((xterm[s] <= piterm[s] for s in states))

    ## Constraints to guarantee monotonicity across states
    # m.addConstrs((x[s, t, a] <= quicksum(x[s + 1, t, aa] for aa in [aa for aa in actions if aa >= a])
    #               for s in [s for s in states if s < max(states)] for t in dec_epochs for a in actions)) # used to work but now gives an error
    for s in [s for s in states if s < max(states)]:
        for t in dec_epochs:
            for a in actions:
                m.addConstr(x[s, t, a] <= quicksum(x[s + 1, t, aa] for aa in [aa for aa in actions if aa >= a]))

    ## Constraints to ensure the policy is feasible
    m.addConstrs((quicksum (y[s, t, ia] for ia in infeasible[s][t]) == 0 for s in states for t in dec_epochs))

    ## Additional constraints to guarantee monotonicity over decision epochs
    # m.addConstrs((x[s, t, a] <= quicksum(x[s, t+1, aa] for aa in [aa for aa in actions if aa >= a])
    #               for s in states for t in dec_epochs[:-1] for a in actions)) # used to work but now gives an error
    for s in states:
        for t in dec_epochs[:-1]:
            for a in actions:
                m.addConstr(x[s, t, a] <= quicksum(x[s, t+1, aa] for aa in [aa for aa in actions if aa >= a]))

    ## Additional constraints to aid solving the MIPs
    m.addConstr(quicksum(r[s, t, a]*y[s, t, a] for s in states for t in dec_epochs for a in actions) +
                quicksum(rterm[s]*xterm[s] for s in states) <= J_opt)

    for s in states:
        for t in dec_epochs:
            m.addSOS(GRB.SOS_TYPE1, [x[s, t, a] for a in actions], list(range(1, A+1)))

    # Processing model specifications
    m.update()

    # Surpressing output
    m.setParam('OutputFlag', False)

    # Setting time limit to 1 hour
    m.setParam('TimeLimit', 3600)

    # Storing only 1 MIP solution
    m.setParam('PoolSolutions', 1)

    # Changing the focus of the MIP solver #use 2 to focus on proving optimality #use 3 to focus on the bound
    m.setParam('MIPFocus', 3)

    # Changing the tolerance level of the MIP
    m.setParam('MIPGapAbs', 0.001) # 0.019 for within 1 week of perfect health #0.0025 for within 1 day of perfect health #0.01 for 1% of a year of perfect health

    # Not pre-solving the model to avoid numerical issues?
    m.setParam('Presolve', 0)

    # Optimizing model
    m.optimize()

    # Extracting objective value and optimal policy
    d_mopt = np.empty((S, T)); d_mopt[:] = np.nan
    if m.Status == 2:  # Model was solved to optimality
        # Storing optimal value of objective function
        J_mopt = m.objVal

        # Extracting decision rule
        for t in epochs:
            for s in states:
                if t < max(epochs):
                    for a in actions:
                        if np.round(x[s, t, a].X) > 0:
                            d_mopt[s, t] = a

        # Evaluating policy
        V_mopt = evaluate_pi(d_mopt, P, r, rterm, gamma)

        # Calculating expected number of events following policy
        e_mopt = evaluate_events(d_mopt, P, event_states)

    else:
        J_mopt = np.nan  # Indicator that the MIP was not solved to optimality
        V_mopt = np.empty((S, T+1)); V_mopt[:] = np.nan  # Indicator that the MIP was not solved to optimality
        e_mopt = np.empty((S, T+1)); e_mopt[:] = np.nan  # Indicator that the MIP was not solved to optimality

    return V_mopt, d_mopt, J_mopt, e_mopt

# Function to evaluate AHA's guidelines
def aha_eval(d_aha, P, r, rterm, alpha, gamma, event_states):

    # Evaluating policy from AHA's guidelines in true transition probabilities
    V_aha = evaluate_pi(d_aha, P, r, rterm, gamma)

    # Calculating total expected discounted reward
    J_aha = np.dot(alpha.flatten(), V_aha.flatten())

    # Calculating expected number of events following policy
    e_aha = evaluate_events(d_aha, P, event_states)

    return V_aha, J_aha, e_aha

# Function to evaluate the no treatment policy
def notrt_eval(P, r, rterm, gamma, event_states):

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs

    # Matrix of representing "no treatment"
    d_notrt = np.zeros((S, T), dtype=int)

    # Evaluating the no treatment policy
    V_notrt = evaluate_pi(d_notrt, P, r, rterm, gamma)

    # Calculating expected number of events following policy
    events = evaluate_events(d_notrt, P, event_states)

    return V_notrt, events

# Traditional Q-learning algorithm (assuming 1/n step-size and epsilon-greedy behavior policy)
def q_learning(P, r, rterm, alpha, gamma, N, epsilon, healthy, feasible, event_states):
    """
    Inputs:
        P: S x S x T x A array of transition probabilities
        r: S x T x A array of rewards
        rterm: array of size S of terminal rewards
        alpha: initial state distribution
        gamma: discount factor between 0 and 1
        N: maximum number of episodes
        epsilon: value of exploration parameter
        healthy: indicator of healthy state
        feasible: indicator of clinically feasible actions
        event_states: indicator of states representing ASCVD events
    """""

    """
    Outputs:
        Q_hat: estimate of action-value functions
        pi: S x A array of approximately optimal probabilities of selecting each action at every state
    """""

    # Extracting parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Initializing parameters
    N_sa = np.zeros((S, T, A))  # matrix to store number of observations in each state and action pair (for step-size)
    seed = 100 # initial seed for pseudo-random number generator
    Q_hat = np.zeros((S, T, A)) # initializing action-value functions
    b = np.ones((S, T, A))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = np.argmax(Q_hat, axis=2) # identifying greedy action in each state and decision epoch

    ## Increasing the probability of selection of the best action in each state and decision epoch
    for s in range(S):
        for t in range(T):
            b[s, t, greedy[s, t]] += (1-epsilon)

    for n in range(N): # each episode
        epsilon = 1/(n + 1) # updating value of epsilon
        # Generating initial state
        s_now = healthy  # assuming patients are healthy at the beginning of the planning horizon

        for t in range(T):  # continue in episode until we reach end of episode (the length of the episodes are determined by the planning horizon)

            # Selecting current action according to behavior policy
            np.random.seed(seed); seed += 1  # establishing seed
            a_now = np.random.choice(np.arange(b.shape[2]), p=b[s_now, t, :])  # selecting next action

            # Determining next state from transition probabilities
            np.random.seed(seed); seed += 1  # establishing seed
            s_next = np.random.choice(np.arange(S), p=P[s_now, :, t, a_now]) # sampling next state

            # Updating estimate of action-value function
            N_sa[s_now, t, a_now] += 1; step_size = 1/N_sa[s_now, t, a_now] # establishing step-size parameter (using the Harmonic series as the step-size)
            if (s_next==6 or s_next==7 or s_next==8 or s_next==9):
                Q_hat[s_now, t, a_now] += step_size*(r[s_now, t, a_now]) - Q_hat[s_now, t, a_now]
                break
            if t == max(range(T)):
                Q_hat[s_now, t, a_now] += step_size * ((r[s_now, t, a_now] + gamma * rterm[s_next]) - Q_hat[s_now, t, a_now])  # updating action-value function with terminal reward
            else:
                np.random.seed(seed); seed += 1  # establishing seed
                a_next = np.argmax(Q_hat[s_next, t + 1, :])  # selecting next action according to greedy policy
                Q_hat[s_now, t, a_now] += step_size*((r[s_now, t, a_now] + gamma * Q_hat[s_next, t + 1, a_next]) - Q_hat[s_now, t, a_now]) # updating action-value function

            # Updating epsilon-greedy policy
            b[s_now, t, :] = np.ones(A)*epsilon/A # assigning epsilon/A probability of selection to all actions
            greedy = np.argmax(Q_hat[s_now, t, :]) # identifying greedy action in current state
            b[s_now, t, greedy] += (1-epsilon) # increasing the probability of selection of the best action in current state

    # Identifying approximately optimal policy (while considering clinical feasibility)
    d_q_learn = np.empty((S, T)); d_q_learn[:] = np.nan
    for s in range(S):
        for t in range(T):
            d_q_learn[s, t] = np.argmax(Q_hat[s, t, feasible[s][t]])

    # Evaluating policy in true transition probabilities
    V_q_learn = evaluate_pi(d_q_learn.astype(int), P, r, rterm, gamma)

    # Calculating total expected discounted reward
    J_q_learn = np.dot(alpha.flatten(), V_q_learn.flatten())

    # Calculating expected number of events following policy
    e_q_learn = evaluate_events(d_q_learn, P, event_states)

    return V_q_learn, d_q_learn, J_q_learn, e_q_learn

# Monotone Q-learning
def update_predecessor(S_now,A_now,record):
    current_state=S_now
    current_action=A_now
    while current_state>0: #while we are not arriving at state 0
        visited=record[current_state][0]
        if visited!=-1:
            record[current_state-1][2]=visited
        if visited==-1:
            record[current_state-1][2]=A_now
        Lowerbound=record[current_state-1][1]
        Upperbound=record[current_state-1][2]
        if Lowerbound>Upperbound:
            record[current_state-1][1]=Upperbound
        current_state-=1
    return record
def update_successor(S_now,A_now,record):
    current_state=S_now
    current_action=A_now
    while current_state<((np.shape(record)[0])-1):
        visited=record[current_state][0]
        if visited!=-1:
            record[current_state+1][1]=visited
        if visited==-1:
            record[current_state+1][1]=record[current_state][1]
        Lowerbound=record[current_state+1][1]
        Upperbound=record[current_state+1][2]
        if Lowerbound>Upperbound:
            record[current_state+1][2]=Lowerbound
        current_state+=1
    return record
def temporal_b(A,epsilon,a_L,a_U,greedy):
    b= np.zeros(A)
    b[greedy]+=(1-(epsilon))
    action_num=len(range(a_L,(a_U+1)))
    b[a_L:(a_U+1)]+=(epsilon/action_num)
    b=b.T/b.sum()
    b=b.T
    return b
def argmax_Q(Q,s_now,T,a_L,a_U):
    Q_snow=Q[s_now,T,:].copy()
    k=np.argmax(Q_snow[a_L:a_U+1])
    return a_L+k
def Monotonic_Q_learning(P, r, rterm, alpha, gamma, N, epsilon, healthy, infeasible, event_states, J_mopt):
    """
    Inputs:
        P: S x S x T x A array of transition probabilities
        r: S x T x A array of rewards
        rterm: array of size S of terminal rewards
        alpha: initial state distribution
        gamma: discount factor between 0 and 1
        N: maximum number of episodes
        epsilon: value of exploration parameter
        healthy: indicator of healthy state [0, 1, 4, 2, 5, 3, |6, 7, 8, 9|]
        feasible: indicator of clinically feasible actions
        event_states: indicator of states representing ASCVD events
    """""

    """
    Outputs:
        Q_hat: estimate of action-value functions
        pi: S x A array of approximately optimal probabilities of selecting each action at every state
    """""

    # Extracting parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Initializing parameters
    N_sa = np.zeros((S, T, A))  # matrix to store number of observations in each state and action pair (for step-size)
    seed = 100 # initial seed for pseudo-random number generator
    Q_hat = np.zeros((S, T, A)) # initializing action-value functions
    b = np.ones((S, T, A))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = np.argmax(Q_hat, axis=2) # identifying greedy action in each state and decision epoch
    b[np.arange(b.shape[0]), greedy] += (1 - epsilon)  # increasing the probability of selection of the best action in each state

    for n in range(N): # each episode
        epsilon = 1 / (n + 1)  # updating value of epsilon
        # Generating initial state
        # record = np.reshape(np.array((-1, 0, A) * 6))  # NEW: LINE REMOVED
        record = np.reshape(np.array((-1, 0, A) * S), (S, 3)) # NEW: LINE ADDED
        s_now = healthy  # assuming patients are healthy at the beginning of the planning horizon
        # b[s_now, 0, greedy[s_now, 0]] += (1-epsilon) # NEW: LINE REMOVED
        # greedy = np.argmax(Q_hat[s_now, t, :])
        for t in range(T):
            a_now = np.random.choice(np.arange(b.shape[2]), p=b[s_now, t, :])
            record[s_now][0] = a_now
            record = update_predecessor(s_now, a_now, record)
            record = update_successor(s_now, a_now, record)
            np.random.seed(seed); seed += 1  # establishing seed
            N_sa[s_now, t, a_now] += 1; step_size = 1/N_sa[s_now, t, a_now]
            s_next = np.random.choice(np.arange(S), p=P[s_now, :, t, a_now]) # sampling next state
            if s_next == 6 or s_next == 7 or s_next == 8 or s_next == 9:
                Q_hat[s_now, t, a_now] += step_size*(r[s_now, t, a_now]) - Q_hat[s_now, t, a_now]
                # print('AAAAAAAAAA')
                break
            if t == max(range(T)):
                Q_hat[s_now, t, a_now] += step_size * ((r[s_now, t, a_now] + gamma * rterm[s_next]) - Q_hat[s_now, t, a_now])
                # print('BBBBBBBBBB')
                break
            np.random.seed(seed); seed += 1
            a_next_lower = record[s_next][1]
            a_next_upper = record[s_next][2]
            if a_next_lower == a_next_upper:#a_next=a_next_lower
                greedy_next = a_next_lower
            else:
                greedy_next = argmax_Q(Q_hat, s_next, t+1, a_next_lower, a_next_upper)
            b[s_next, t+1, :] = temporal_b(A, epsilon, a_next_lower, a_next_upper, greedy_next) # NEW: s_now, WAS REMOVED AS PARAMETER
            #a_next =np.random.choice(np.arange(b.shape[2]), p=b[s_next,t+1, :])
            Q_hat[s_now, t, a_now] += step_size*((r[s_now, t, a_now] + gamma * Q_hat[s_next, t + 1, greedy_next]) - Q_hat[s_now, t, a_now])
            s_now = s_next
            #a_now=a_next

    # Generating list of tuples of state-action pairs
    states = list(np.arange(0, S, 1))
    actions = list(np.arange(0, A, 1))
    dec_epochs = list(np.arange(0, T, 1))
    state_action = []
    for s in states:
        for t in dec_epochs:
            for a in actions:
                state_action.append((s, t, a))

    # Creating Gurobi model object
    m = Model()

    # Adding decision variables to model
    d = m.addVars(state_action, vtype=GRB.BINARY)

    # Declaring model objective
    m.setObjective(quicksum(Q_hat[s, t, a]*d[s, t, a] for s in states for t in dec_epochs for a in actions), GRB.MAXIMIZE)

    # Adding constraints
    m.addConstrs((quicksum(d[s, t, a] for a in actions) == 1 for s in states for t in dec_epochs))
    for s in [s for s in states if s < max(states)]:
        for t in dec_epochs:
            for a in actions:
                m.addConstr(d[s, t, a] <= quicksum(d[s + 1, t, aa] for aa in [aa for aa in actions if aa >= a]))
    for s in states:
        for t in dec_epochs[:-1]:
            for a in actions:
                m.addConstr(d[s, t, a] <= quicksum(d[s, t+1, aa] for aa in [aa for aa in actions if aa >= a]))
    for s in states:
        for t in dec_epochs:
            m.addSOS(GRB.SOS_TYPE1, [d[s, t, a] for a in actions], list(range(1, A+1)))
    m.addConstrs((quicksum(d[s, t, ia] for ia in infeasible[s][t]) == 0 for s in states for t in dec_epochs))
    # m.addConstr(quicksum(Q_hat[s, t, a]*d[s, t, a] for s in states for t in dec_epochs for a in actions) <= J_mopt) # NEW: LINE REMOVED (THIS CONDITION IS NOT EQUIVALENT TO BOUNDING THE WEIGHTED SUM OF REWARDS BY THE MONOTONE OPTIMAL OBJECTIVE VALUE)

    # Processing model specifications
    m.update()

    # Surpressing output
    m.setParam('OutputFlag', False)

    # Setting time limit to 1 hour
    m.setParam('TimeLimit', 3600)

    # Storing only 1 MIP solution
    m.setParam('PoolSolutions', 1)

    # Changing the focus of the MIP solver #use 2 to focus on proving optimality #use 3 to focus on the bound
    m.setParam('MIPFocus', 3)

    # Changing the tolerance level of the MIP
    m.setParam('MIPGapAbs', 0.01) # 0.019 for within 1 week of perfect health #0.0025 for within 1 day of perfect health #0.01 for 1% of a year of perfect health

    # Optimizing model
    m.optimize()

    # Extracting objective value, optimal value function, and optimal policy
    d_mlearn = np.empty((S, T)); d_mlearn[:] = np.nan
    if m.Status == 2:  # Model was solved to optimality
        # Storing optimal value of objective function
        # J_mlearn = m.objVal # NEW: LINE REMOVED

        # Extracting decision rule
        for t in dec_epochs:
            for s in states:
                # if t < max(dec_epochs): # NEW: LINE REMOVED (HYPERTENSION TREATMENT PROCESS IS MODELED SO THAT THERE IS AN ACTION AT THE LAST STAGE)
                for a in actions:
                    if np.round(d[s, t, a].X) > 0:
                        d_mlearn[s, t] = a

        # Evaluating policy
        V_mlearn = evaluate_pi(d_mlearn.astype(int), P, r, rterm, gamma)

        # Calculating total expected discounted reward
        J_mlearn = np.dot(alpha.flatten(), V_mlearn.flatten())

        # Calculating expected number of events following policy
        e_mlearn = evaluate_events(d_mlearn.astype(int), P, event_states)

        # Calculating expected number of events following policy
    else: # Display warning message and do not store results (model was not solved to optimality)
        # print("Monotone MDP in states and decision epochs was not solved to optimality. Status code: ", m.Status)
        # if hasattr(m, 'ObjVal'):
        #     print("Final absolute MIP gap value: ", float(m.MIPGap)*abs(m.ObjVal))
        # else:
        #     print("Final absolute MIP gap value: ", "NA")
        J_mlearn = np.nan  # Indicator that the MIP was not solved to optimality
        V_mlearn = np.empty((S, T+1)); V_mlearn[:] = np.nan  # Indicator that the MIP was not solved to optimality
        e_mlearn = np.empty((S, T+1)); e_mlearn[:] = np.nan  # Indicator that the MIP was not solved to optimality
    #pi=pi.astype(int)
    #print(V_mlearn, d_mlearn, J_mlearn, e_mlearn)
    return V_mlearn, d_mlearn, J_mlearn, e_mlearn
