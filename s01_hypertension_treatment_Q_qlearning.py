# ******************************************************

# Monotone Policies Case Study - Hypertension Treatment
# ******************************************************

# Notes:
# 1. Script revised from hypertension_treatment_monotone_mdp.py in the "Monotone Policies" folder of the code archive
# 2. Path to files must be updated before the script is run
# 3. Code to obtain results presented in "Interpretable Policies and the Price of Interpretability in Hypertension
# Treatment Planning" are commented but kept in script for reproducibility purposes.
# 4. Areas in the script that must be updated can be found by searching for "#####" (no quotes)

# To do:
# 1. Add monotone q-learning function in s06_optimal_monotone_mdp.py and call function in s03_patient_simulation.py
# 2. Run small batch of patients to examine results
# 3. Update plot_policies_state function in s10_case_study_plots.py with new results
# 4. Run complete set of patients to examine convergence (maximum difference in expected total discounted reward between monotone Q-learning and monotone optimal policy?)
# 5. Plot policy results for selected patient profiles (Figure 1 in MSOM paper), QALYs saved in healthy state at year 1 (able 1 in MSOM paper), regret-adjusted price of interpretability?

# Loading modules
import os  # directory changes
import pandas as pd  # data frame operations
import numpy as np  # array operations
import time as tm  # timing code
import itertools  # recursive operations (for sequential execution)
import multiprocessing as mp  # parallel computations
import pickle as pk  # saving results
from s02_bp_med_effects import med_effects # medication parameters (using a generic drug type)
import s03_patient_simulation as pt_sim # risk, transition probabilities, and policy calculations

# Establishing directories (Change to appropriate directory)
if os.name == 'posix': # name for linux system (for Dartmouth Babylons)
    home_dir = os.path.abspath(os.environ['HOME'] + '/Documents/Monotone Policies/Python')
    data_dir = os.path.abspath(os.environ['HOME'] + '/Documents/Monotone Policies/Data')
    results_dir = os.path.abspath(os.environ['HOME'] + '/Documents/Monotone Policies/Python/Results')
    sens_dir = os.path.abspath(os.environ['HOME']+'/Documents/Monotone Policies/Python/Results/Sensitivity Analyses')
    fig_dir = os.path.abspath(os.environ['HOME']+'/Documents/Monotone Policies/Python/Figures')
else:
    dir_path = os.path.abspath(os.environ['USERPROFILE'] + '\\My Drive\\Research\\Research Group\\''Current Students\\Mina Yi\\Chapter 1\\WSC 2024')  # Overall path to directories
    home_dir = os.path.abspath(dir_path + '\\PythonProject1')
    data_dir = os.path.abspath(dir_path + '\\Data')
    results_dir = os.path.abspath(home_dir + '\\Results')
    fig_dir = os.path.abspath(home_dir + '\\Figures')

# =============
# Loading data
# =============

# Loading life expectancy and death likelihood data
# (first column age, second column male, third column female)
os.chdir(data_dir)
#abcdefd=os.getcwd()
#print(abcdefd)
lifedata = pd.read_csv('lifedata.csv', header=None)
strokedeathdata = pd.read_csv('strokedeathdata.csv', header=None)
chddeathdata = pd.read_csv('chddeathdata.csv', header=None)
alldeathdata = pd.read_csv('alldeathdata.csv', header=None)

# Loading risk slopes (first column age, second column CHD, third column stroke)
riskslopedata = pd.read_csv('riskslopes.csv', header=None)

# Loading 2009-2016 Continuous NHANES dataset (ages 40-60)
if os.name == 'posix': # name for linux system (for Dartmouth Babylons)
    os.chdir(data_dir + '/Continuous NHANES')
else:
    os.chdir(data_dir+'\\Continuous NHANES')
ptdata = pd.read_csv('Continuous NHANES Forecasted Dataset.csv') # using sampling weights as recorded in the NHANES dataset (to reduce computational burden)

# =======================
# Initializing parameters
# =======================

# Selecting number of cores for parallel processing
if os.name == 'posix': # name for linux system (for Dartmouth Babylons)
    cores = 35
else:
    cores = mp.cpu_count() - 1

# Risk parameters
ascvd_hist_mult = [3, 3]  # multiplier to account for history of CHD and stroke, respectively

# Transition probability parameters
numhealth = 10  # Number of health states
years = 10  # Number of years (non-stationary stages)
events = 2  # Number of events considered in model

# Treatment parameters
## BP clinical constraints
sbpmin = 120  # minimum allowable SBP
dbpmin = 55  # minimum allowable DBP
sbpmax = 150  # maximum allowable untreated SBP
dbpmax = 90  # maximum allowable untreated DBP

## AHA's guideline parameters
targetrisk = 0.1
targetsbp = 130
targetdbp = 80

## Half dosages compared to standard dosages
hf_red_frac = 2/3 # fraction of BP and risk reduction
hf_disut_frac = 1/2 # fraction of disutility

## Estimated change in BP by dosage (assuming absolute BP reductions and linear reductions with respect to dose)
sbp_drop_std = 5.5 # average SBP reduction per medication at standard dose in BPLTCC trials
sbp_drop_hf = sbp_drop_std*hf_red_frac # average SBP reduction per medication at half dose
dbp_drop_std = 3.3 # average DBP reduction per medication at standard dose in BPLTCC trials
dbp_drop_hf = dbp_drop_std*hf_red_frac # average DBP reduction per medication at half dose

# Estimated change in risk by dosage (assuming absolute risk reductions)
rel_risk_chd_std = 0.87 # estimated change in risk for CHD events per medication at standard dose in BPLTCC trials
rel_risk_stroke_std = 0.79 # estimated change in risk for stroke events per medication at standard dose in BPLTCC trials
rel_risk_chd_hf = 1-((1-rel_risk_chd_std)*hf_red_frac) # estimated change in risk for CHD events per medication at half dose
rel_risk_stroke_hf = 1-((1-rel_risk_stroke_std)*hf_red_frac) # estimated change in risk for stroke events per medication at half dose

## Estimated treatment disutility by dosage
disut_std = 0.002 # treatment disutility per medication at standard dose
disut_hf = disut_std*hf_disut_frac # treatment disutility per medication at half dose

## Treatment choices (21 trts: no treatment plus 1 to 5 drugs at standard and half dosages)
allmeds = list(range(21))  # index for possible treatment options
numtrt = len(allmeds)  # number of treatment choices

## Treatment effects (SBP reductions, DBP reductions, post-treatment relative risk for CHD events,
# post-treatment relative risk for stroke events, and treatment related disutilities)
sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, trtharm, meds = med_effects(hf_red_frac, sbp_drop_std,
                                                                                         sbp_drop_hf, dbp_drop_std,
                                                                                         dbp_drop_hf, rel_risk_chd_std,
                                                                                         rel_risk_chd_hf, rel_risk_stroke_std,
                                                                                         rel_risk_stroke_hf, disut_std,
                                                                                         disut_hf, numtrt,
                                                                                         "nondecreasing") # "nonincreasing"

std_index = [0, 2, 6, 11, 17, 20] # index of standard doses

# MDP parameters
## Discounting factor
gamma = 0.97

# Immediate rewards - QALY parameters (Kohli-Lynch et al. 2019)
QoL = {"40-44": [1, 0.9348, 0.8835, 0.9348*0.8835, 0.8970*(1/12)+0.9348*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
       "45-54": [1, 0.9374, 0.8835, 0.9374*0.8835, 0.8862*(1/12)+0.9374*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
       "55-64": [1, 0.9376, 0.8835, 0.9376*0.8835, 0.8669*(1/12)+0.9376*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
       "65-74": [1, 0.9372, 0.8835, 0.9372*0.8835, 0.8351*(1/12)+0.9372*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
       "75-84": [1, 0.9364, 0.8835, 0.9363*0.8835, 0.7946*(1/12)+0.9363*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
       }

## Terminal reward
### Terminal QoL weights
QoLterm = {"40-44": [1, 0.9348, 0.8835, 0.9348*0.8835, 0.9348, 0.8835, 0, 0, 0, 0],
           "45-54": [1, 0.9374, 0.8835, 0.9374*0.8835, 0.9374, 0.8835, 0, 0, 0, 0],
           "55-64": [1, 0.9376, 0.8835, 0.9376*0.8835, 0.9376, 0.8835, 0, 0, 0, 0],
           "65-74": [1, 0.9372, 0.8835, 0.9372*0.8835, 0.9372, 0.8835, 0, 0, 0, 0],
           "75-84": [1, 0.9364, 0.8835, 0.9363*0.8835, 0.9364, 0.8835, 0, 0, 0, 0]
           }

### Terminal condition standardized mortality rate (first list males, second list females)
mortality_rates = {"Males <2 CHD events":    [1, 1/1.6, 1/2.3, (1/1.6)*(1/2.3), 1/1.6, 1/2.3, 0, 0, 0, 0],
                   "Females <2 CHD events":  [1, 1/2.1, 1/2.3, (1/2.1)*(1/2.3), 1/2.1, 1/2.3, 0, 0, 0, 0],
                   "Males >=2 CHD events":   [1, 1/3.4, 1/2.3, (1/3.4)*(1/2.3), 1/3.4, 1/2.3, 0, 0, 0, 0],
                   "Females >=2 CHD events": [1, 1/2.5, 1/2.3, (1/2.5)*(1/2.3), 1/2.5, 1/2.3, 0, 0, 0, 0]
                   }

# State order (nonincreasing order of rewards)
## Order: (0) healthy, (1) history of CHD, (2) history of stroke, (3) history of CHD and stroke, (4) surviving a CHD,
# (5) surviving a stroke, (6) dying from non-ASCVD related cause, (7) dying from a CHD, (8) dying from a stroke, and (9) death
state_order = [0, 1, 4, 2, 5, 3, 6, 7, 8, 9]

# Identification of states
healthy = int(np.squeeze(np.where(np.array(state_order) == 0))) # states at which the patient has not experienced ASCVD events
alive = np.where([j in range(6) for j in state_order])[0] # states at which the patient is alive
chd_hist = np.where([j in [1, 3, 4] for j in state_order])[0] # states where chd risk is higher (order not changed for risk calculations to avoid numerical errors in transition probabilities)
stroke_hist = np.where([j in [2, 3, 5] for j in state_order])[0]  # states where stroke risk is higher (order not changed for risk calculations to avoid numerical errors in transition probabilities)
dead = np.where([j in range(6, 10) for j in state_order])[0] # states at which the patient is dead

## Indicators of states where patients are experiencing ASCVD events
event_states = np.zeros(numhealth) # vector of zeros
event_states[np.where([j in [4, 5, 7, 8] for j in state_order])[0]] = 1 # replacing appropiate indexes with 1
event_states = event_states.astype(int).tolist() # indicators of states where patients are experiencing ASCVD events

## Scaling odds to account for history of adverse events
ascvd_hist_scale = np.ones([numhealth, events]) # array of ones
ascvd_hist_scale[chd_hist, 0] = ascvd_hist_mult[0]  # Attaching CHD multiplier
ascvd_hist_scale[stroke_hist, 1] = ascvd_hist_mult[1]  # Attaching stroke multiplier
ascvd_hist_scale[dead, :] = 0  # risk becomes 0 if the patient is not alive

# State distribution
alpha = np.zeros((numhealth, (years+1))) # matrix of zeros for sensitivity analysis added after 1st revision
alpha[healthy, 0] = 1 # all weight focused on year 1 and healthy condition - weight according to NHANES

# Q-learning parameters
N = int(10) # number of iterations (chosen arbritrarily, according to common practices)
epsilon = 1 # initial exploration parameter


# ==================
# Patient simulation
# ==================

# # Evaluating different patient profiles
# ## Sampling patient
# sample_pt = ptdata[ptdata.id == 472].copy() # random patient in data set with elevated BP levels
# sample_pt.tc -= 20 # making sure the profile has normal total cholesterol levels
# sample_pt.ldl -= 40 # making sure the profile has normal LDL
# sample_pt.sbp -= 0.5 # making sure the profile has elevated BP levels
# sample_pt.sex = 1 # chainging the profile's sex to male
#
# ## Adding modified versions of the patient
# ### Normal BP
# tmp = sample_pt.copy(); tmp.sbp -= 10
# sample_df = pd.concat([tmp, sample_pt], axis=0).reset_index(drop=True, inplace=False)
#
# ### Stage 1 hypertension
# tmp = sample_pt.copy(); tmp.sbp += 5; tmp.dbp += 5
# sample_df = pd.concat([sample_df, tmp], axis=0).reset_index(drop=True, inplace=False)
#
# ### Stage 2 hypertension
# tmp = sample_pt.copy(); tmp.sbp += 20; tmp.dbp += 5
# sample_df = pd.concat([sample_df, tmp], axis=0).reset_index(drop=True, inplace=False)
#
# ### Modifying ids
# sample_df.id = pd.Series(np.repeat(np.arange(10), years))
# del sample_pt # deleting sample patient
# print(sample_df.round())

os.chdir(home_dir)
keys = ('pt_id', # 'V_notrt', 'e_notrt', # 'risk', 'P',
        # 'V_opt', 'd_opt', 'J_opt', 'e_opt',
        # 'V_mopt', 'd_mopt', 'J_mopt', 'e_mopt',
        # 'V_aha', 'd_aha', 'J_aha', 'e_aha',
        # 'V_q_learn', 'd_q_learn', 'J_q_learn', 'e_q_learn',
        'V_mlearn', 'd_mlearn', 'J_mlearn', 'e_mlearn'
        ) # keys for dictionary to store results

pt_ids = range(500) # Run 500 patients here. [539, 472, 1116, 1660] # range(100) # randomly chosen records # range(len(ptdata.id.unique())) # all patient ids # [539, 472, 1116, 1660] # patient profiles
# ptdata.loc[ptdata.id == 1660, 'sbp'] += 5 # making sure the last profile has worse health condition than the second to last

if __name__ == '__main__': # Ensuring the code is run as the main module	
    # Running patient simulation in parallel
    ## Running time note:
    start_time_par = tm.time()
    with mp.Pool(cores) as pool:  # Creating pool of parallel workers
        results = pool.starmap_async(pt_sim.patient_sim, [(i, ptdata[ptdata.id == i], # sample_df for patient profiles # ptdata for population
                                                           numhealth, healthy, dead, years, events,
                                                           stroke_hist, ascvd_hist_scale, event_states,
                                                           lifedata, mortality_rates,
                                                           chddeathdata, strokedeathdata, alldeathdata,
                                                           sbpmin, dbpmin, sbpmax, dbpmax, allmeds, trtharm,
                                                           sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke,
                                                           QoL, QoLterm, alpha, gamma, state_order, N, epsilon,
                                                           targetrisk, targetsbp, targetdbp)
                                                          for i in pt_ids]).get()
    end_time_par = tm.time()
    print("--- %s minutes ---" % ((end_time_par - start_time_par) / 60))
    # print('len(par_results)', len(par_results[0]))

    # # Running patient simulation sequentially
    # start_time = tm.time()
    # results = list(itertools.starmap(pt_sim.patient_sim, [(i, ptdata[ptdata.id == i],
    #                                                        numhealth, healthy, dead, years, events,
    #                                                        stroke_hist, ascvd_hist_scale, event_states,
    #                                                        lifedata, mortality_rates,
    #                                                        chddeathdata, strokedeathdata, alldeathdata,
    #                                                        sbpmin, dbpmin, sbpmax, dbpmax, allmeds, trtharm,
    #                                                        sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke,
    #                                                        QoL, QoLterm, alpha, gamma, state_order, N, epsilon,
    #                                                        targetrisk, targetsbp, targetdbp)
    #                                                       for i in pt_ids]))
    # end_time = tm.time()
    # print("--- %s minutes ---" % ((end_time-start_time)/60))

    ## Storing results in a dictionary
    values = ([d[res] for d in results] for res in range(len(results[0])))
    ptresults = dict(zip(keys, values))

    # Removing dead states (kept for debugging purposes)
    keys_to_extract = [# 'V_notrt', 'e_notrt'
        # 'V_opt', 'd_opt', 'e_opt',
        # 'V_mopt', 'd_mopt', 'e_mopt',
        # 'V_aha', 'd_aha', 'e_aha',
        # 'V_q_learn', 'd_q_learn', 'e_q_learn',
        'V_mlearn', 'd_mlearn', 'e_mlearn'
    ]
    tmp_dict = {k: [np.delete(x, dead, axis=0) for x in ptresults[k]] for k in keys_to_extract}
    ptresults.update(tmp_dict); del tmp_dict

    # # Extracting dictionary values to local environment (for debugging purposes)
    # locals().update(ptresults)

    # Saving results
    os.chdir(home_dir)
    if not os.path.isdir(results_dir):
        os.mkdir("Results")
    os.chdir(results_dir)
    with open('Results for ' + str(len(pt_ids)) + ' patients with ' + str(N) + ' learning iterations.pkl', 'wb') as f:
        pk.dump(ptresults, f, protocol=3)
