# =======================================================
# Summary of Results - Hypertension treatment case study
# =======================================================

# Loading modules
import os  # directory changes
import numpy as np  # array operations
import itertools as it # iterative computations
import pandas as pd  # data frame operations
import pickle as pk  # saving results
import gc # clearing memory
from s10_case_study_plots import plot_policies_state # plot functions

# Importing parameters from main module
from s01_hypertension_treatment_monotone_qlearning import results_dir, fig_dir, ptdata, years, alpha, alive, meds

# Presenting results using per capita rate
per_capita = 100000 # results per per_capita number of patients

# Extracting first year data
ptdata1 = ptdata.groupby('id').first().reset_index(drop=False, inplace=False)
del ptdata; gc.collect()

## Estimating mean arterial pressure
ptdata1['map'] = ptdata1.apply(lambda row: (1/3)*(row['sbp']+2*row['dbp']), axis=1)

## Identifying BP categories
bp_cats = [(ptdata1.sbp < 120) & (ptdata1.dbp < 80),
            (ptdata1.sbp >= 120) & (ptdata1.dbp < 80) & (ptdata1.sbp < 130),
            ((ptdata1.sbp >= 130) | (ptdata1.dbp >= 80)) & ((ptdata1.sbp < 140) & (ptdata1.dbp < 90)),
            (ptdata1.sbp >= 140) | (ptdata1.dbp >= 90)]
bp_cat_labels = ['Normal BP', 'Elevated BP', 'Stage 1 Hypertension', 'Stage 2 Hypertension'] # Hypertension
ptdata1['bp_cat'] = np.select(bp_cats, bp_cat_labels)

# ---------------------------
# Loading and cleaning data
# ---------------------------

# # Loading results of MSOM paper (to avoid repeating the same calculations as in the previous paper)
# os.chdir(results_dir)
# with open('MSOM Paper Results - 4590 patients with 1 hour time limit and 0.001 absolute MIP gap.pkl',
#           'rb') as f:
#
# # Loading new results # uncomment once new results are available
# os.chdir(results_dir)
# with open('Results for 4590 patients with 10000 learning iterations.pkl',
#           'rb') as f:
#     pt_sim_n = pk.load(f)
#
# # Combining dictionaries
# pt_sim = pt_sim | pt_sim_n; del pt_sim_n

# # Using small subset for initial results (use only for creating graphs and debugging)
# tmp_dict = {k: v[:50] for k, v in pt_sim.items()}
# pt_sim.update(tmp_dict); del tmp_dict; gc.collect()

# # Removing unnecessary keys
# rem_list = ['occup', 'V_class_mopt_epochs', 'd_class_mopt_epochs', 'J_class_mopt_epochs', 'e_class_mopt_epochs',
#             'V_risk', 'd_risk', 'J_risk', 'e_risk'] # list of keys to remove
# [pt_sim.pop(key) for key in rem_list]
#
# # Renaming keys to match variable names in analysis
# ## Extracting keys to be renamed
# oldkeys = ['V_mopt_epochs', 'd_mopt_epochs', 'J_mopt_epochs', 'e_mopt_epochs'] # list of keys to rename
# tmp = {k: pt_sim[k] for k in oldkeys} # extracting keys into a new dictionary
#
# ## Renaming keys
# newkeys = ['V_mopt', 'd_mopt', 'J_mopt', 'e_mopt'] # list of new names
# tmp1 = dict(zip(newkeys, list(tmp.values()))) # renaming keys
#
# ## Removing old keys and adding new
# [pt_sim.pop(key) for key in oldkeys] # removing old keys
# pt_sim = pt_sim | tmp1; del tmp, tmp1 # adding new keys and deleting unnecessary variables
#
# # Adding placehoders for Q-learning and monotone Q-learning results
# phkeys = ['V_opt', 'd_opt', 'J_opt', 'e_opt'] # list of keys to rename
# tmp2 = {k: pt_sim[k] for k in phkeys} # extracting keys into a new dictionary
#
# ## Q-learning
# phkeys1 = ['V_q_learn', 'd_q_learn', 'J_q_learn', 'e_q_learn'] # list of new names
# tmp3 = dict(zip(phkeys1, list(tmp2.values()))) # renaming keys
#
# # Monotone Q-learning
# phkeys2 = ['V_mlearn', 'd_mlearn', 'J_mlearn', 'e_mlearn'] # list of new names
# tmp4 = dict(zip(phkeys2, list(tmp2.values()))) # renaming keys
#
# ## Adding placeholder dictionary keys
# pt_sim = pt_sim | tmp3 | tmp4
#
# del phkeys, phkeys1, phkeys2, tmp2, tmp3, tmp4
#
#
# # Excluding NaN values (patients that the MIP was not able to find a solution in 1 hour)
# # Indexes of patients with incomplete policies
# nan_index = np.where([np.isnan(x).any() for x in pt_sim['V_mopt']])[0]
#
# ## Indexes of patients with MIP solutions in all policies
# not_nan_index = np.delete(np.arange(len(pt_sim['pt_id'])), nan_index)
#
# ## Keeping only patients with all policies
# pt_sim = {k: [v[i] for i in not_nan_index] for k, v in pt_sim.items()}

# # Making sure that equal policies do not result in different total discounted reward (to avoid numerical issues in Gurobi)
# tmp_mopt = []
# for y, x in enumerate(pt_sim['d_opt']):
#     ## Making sure total discounted reward of OP is at least the total discounted reward of MP
#     if np.all(x == pt_sim['d_mopt'][y]):
#         tmp_mopt.append(pt_sim['J_opt'][y])
#     else:
#         tmp_mopt.append(pt_sim['J_mopt'][y])
# tmp_dict = {'J_mopt': tmp_mopt}
# pt_sim.update(tmp_dict); del tmp_dict, tmp_mopt # updating dictionary
#
# tmp_mlearn = []
# for y, x in enumerate(pt_sim['d_mopt']):
#     ## Making sure total discounted reward of OP is at least the total discounted reward of MP
#     if np.all(x == pt_sim['d_mlearn'][y]):
#         tmp_mlearn.append(pt_sim['J_mopt'][y])
#     else:
#         tmp_mlearn.append(pt_sim['J_mlearn'][y])
# tmp_dict = {'J_mlearn': tmp_mlearn}
# pt_sim.update(tmp_dict); del tmp_dict, tmp_mlearn # updating dictionary
#
# # Making sure that the total expected reward is OP >= MP (to avoid numerical issues in Gurobi)
# tmp_mopt = []
# for y, x in enumerate(pt_sim['J_opt']):
#
#     ## Making sure the total discounted reward of MP is at most the total discounted reward of OP
#     if x < pt_sim['J_mopt'][y]:
#         tmp_mopt.append(x)
#     else:
#         tmp_mopt.append(pt_sim['J_mopt'][y])
# tmp_dict = {'J_mopt': tmp_mopt}
# pt_sim.update(tmp_dict); del tmp_dict, tmp_mopt # updating dictionary
#
# # Making sure that the total expected reward is MP >= MQ (to avoid numerical issues in Gurobi)
# tmp_mlearn = []
# for y, x in enumerate(pt_sim['J_mopt']):
#
#     ## Making sure the total discounted reward of MQ is at most the total discounted reward of MP
#     if x < pt_sim['J_mlearn'][y]:
#         tmp_mlearn.append(x)
#     else:
#         tmp_mlearn.append(pt_sim['J_mlearn'][y])
# tmp_dict = {'J_mlearn': tmp_mlearn}
# pt_sim.update(tmp_dict); del tmp_dict, tmp_mlearn # updating dictionary

# # ---------------------------
# # Population-level analysis
# # ---------------------------
#
# # Evaluating QALYs saved and events averted per policy, compared to no treatment (in per_capita number of patients)
# polnames = ['Optimal', 'Optimal Monotone', 'Q-learning', 'Monotone Q-learning', 'Clinical Guidelines'] # names of policies
#
# ## Data frames of expected QALYs saved at year 0 and healthy state
# tmp = [[y[0, 0] - pt_sim['V_notrt'][i][0, 0] for i, y in enumerate(pt_sim[x])] for x in ['V_opt', 'V_mopt', 'V_q_learn', 'V_mlearn', 'V_aha']] # calculating expected number of QALYs obtained by each policy in healthy state at the first year of analysis
# tmp1 = pd.DataFrame(tmp, index=polnames).T; del tmp # creating data frame from list of lists
# qalys_df = pd.concat([ptdata1.loc[ptdata1.id.isin(pt_sim['pt_id']), ['id', 'wt', 'bp_cat']].reset_index(drop=True), tmp1], axis=1); del tmp1 # creating data frames with ids, weights, and BP categories
#
# ### Preparing data for summaries
# qalys_df = qalys_df.melt(id_vars=['id', 'wt', 'bp_cat'], var_name='policy', value_name='qalys') # melting data frame
# qalys_df['qalys'] = qalys_df['qalys'].multiply(qalys_df['wt'], axis=0) # adjusting value functions with sampling weights
#
# ### Overall
# tmp = qalys_df.groupby(['policy']).wt.sum() # adding up sampling weights
# qalys_df_ovr = qalys_df.copy() # generating working copy of qalys_df
#
# #### Sorting data frame
# qalys_df_ovr['policy'] = qalys_df_ovr['policy'].astype('category') # converting policy to category
# qalys_df_ovr['policy'] = qalys_df_ovr.policy.cat.set_categories(polnames) # adding sorted categories
# qalys_df_ovr = qalys_df_ovr.sort_values(['policy']) # sorting dataframe based on selected columns
#
# #### Preparing summary of results
# qalys_df_ovr_sum = qalys_df_ovr.groupby(['policy'], observed=False).qalys.sum() # summary of QALYs saved, compared to no treatment
# qalys_df_ovr_sum = qalys_df_ovr_sum.to_frame().join(tmp); del tmp # merging series
# print(((qalys_df_ovr_sum.qalys-qalys_df_ovr_sum.qalys.iloc[-1])/qalys_df_ovr_sum.wt*per_capita).round(2).to_numpy()) # printing QALYs saved per_capita number of patients, compared to the clinical guidelines
# print(((qalys_df_ovr_sum.qalys-qalys_df_ovr_sum.qalys.iloc[-1])/qalys_df_ovr_sum.qalys.iloc[-1]).round(4).to_numpy()) # printing percentage change in QALYs saved over no treatment, compared to the clinical guidelines
#
# ### By BP group
# tmp2 = qalys_df.groupby(['bp_cat', 'policy'], observed=False).wt.sum() # adding up sampling weights
# qalys_df_bp = qalys_df.copy() # generating working copy of qalys_df
#
# #### Sorting data frame
# qalys_df_bp['bp_cat'] = qalys_df_bp['bp_cat'].astype('category') # converting BP categories to category
# qalys_df_bp['bp_cat'] = qalys_df_bp.bp_cat.cat.set_categories(bp_cat_labels) # adding sorted categories
# qalys_df_bp['policy'] = qalys_df_bp['policy'].astype('category') # converting policy to category
# qalys_df_bp['policy'] = qalys_df_bp.policy.cat.set_categories(polnames) # adding sorted categories
# qalys_df_bp = qalys_df_bp.sort_values(['bp_cat', 'policy']) # sorting dataframe based on selected columns
#
# #### Preparing summary of results
# qalys_df_bp_sum = qalys_df_bp.groupby(['bp_cat', 'policy'], observed=False).qalys.sum() # summary of QALYs saved, compared to no treatment
# qalys_df_bp_sum = qalys_df_bp_sum.to_frame().join(tmp2); del tmp2 # merging series
# tmp = np.repeat(qalys_df_bp_sum.loc[(slice(None), 'Clinical Guidelines'), 'qalys'], len(polnames)) # subsetting values corresponding to the monotone policies in multiindex and repeating them the number of policies
# print(((qalys_df_bp_sum.qalys-tmp.to_numpy())/qalys_df_bp_sum.wt*per_capita).round(2).to_numpy()) # printing QALYs saved per_capita number of patients, compared to the clinical guidelines
# print(((qalys_df_bp_sum.qalys-tmp.to_numpy())/tmp.to_numpy()).round(4).to_numpy()) # printing percentage change in QALYs saved over no treatment, compared to the clinical guidelines
#
# # ##### Plotting results
# # os.chdir(fig_dir)
# # qalys_df_bp = qalys_df_bp[qalys_df_bp.bp_cat!=bp_cat_labels[0]] # removing normal BP category
# # qalys_df_bp['bp_cat'] = qalys_df_bp.bp_cat.cat.set_categories(bp_cat_labels[1:]) # adding sorted categories
# # qalys_events(qalys_df_bp)
#
# ## Data frames of expected events averted at year 0 and healthy state
# tmp = [[pt_sim['e_notrt'][i][0, 0] - y[0, 0] for i, y in enumerate(pt_sim[x])] for x in ['e_opt', 'e_mopt', 'e_q_learn', 'e_mlearn', 'e_aha']] # calculating expected number of events allowed by each policy in healthy state at the first year of analysis
# tmp1 = pd.DataFrame(tmp, index=polnames).T; del tmp # creating data frame from list of lists
# events_df = pd.concat([ptdata1.loc[ptdata1.id.isin(pt_sim['pt_id']), ['id', 'wt', 'bp_cat']].reset_index(drop=True), tmp1], axis=1); del tmp1 # creating data frames with ids, weights, and BP categories
# events_df = events_df.melt(id_vars=['id', 'wt', 'bp_cat'], var_name='policy', value_name='events') # melting data frame
# events_df['events'] = events_df['events'].multiply(events_df['wt'], axis=0) # adjusting value functions with sampling weights
#
# ### Overall
# tmp = events_df.groupby(['policy'], observed=False).wt.sum() # adding up sampling weights
# events_df_ovr = events_df.copy() # generating working copy of events_df
#
# #### Sorting data frame
# events_df_ovr['policy'] = events_df_ovr['policy'].astype('category') # converting policy to category
# events_df_ovr['policy'] = events_df_ovr.policy.cat.set_categories(polnames) # adding sorted categories
# events_df_ovr = events_df_ovr.sort_values(['policy']) # sorting dataframe based on selected columns
#
# #### Preparing summary of results
# events_df_ovr = events_df_ovr.groupby(['policy'], observed=False).events.sum() # summary of events saved, compared to no treatment
# events_df_ovr = events_df_ovr.to_frame().join(tmp); del tmp # merging series
# print(((events_df_ovr.events-events_df_ovr.events.iloc[-1])/events_df_ovr.wt*per_capita).round(2).to_numpy()) # printing events averted per_capita number of patients, compared to the clinical guidelines
# print(((events_df_ovr.events-events_df_ovr.events.iloc[-1])/events_df_ovr.events.iloc[-1]).round(4).to_numpy()) # printing percentage change in events saved over no treatment, compared to the clinical guidelines
#
# ### By BP group
# tmp2 = events_df.groupby(['bp_cat', 'policy'], observed=False).wt.sum() # adding up sampling weights
# events_df_bp = events_df.copy() # generating working copy of events_df
#
# #### Sorting data frame
# events_df_bp['bp_cat'] = events_df_bp['bp_cat'].astype('category') # converting BP categories to category
# events_df_bp['bp_cat'] = events_df_bp.bp_cat.cat.set_categories(bp_cat_labels) # adding sorted categories
# events_df_bp['policy'] = events_df_bp['policy'].astype('category') # converting policy to category
# events_df_bp['policy'] = events_df_bp.policy.cat.set_categories(polnames) # adding sorted categories
# events_df_bp = events_df_bp.sort_values(['bp_cat', 'policy']) # sorting dataframe based on selected columns
#
# #### Preparing summary of results
# events_df_bp_sum = events_df_bp.groupby(['bp_cat', 'policy'], observed=False).events.sum() # summary of events saved, compared to no treatment
# events_df_bp_sum = events_df_bp_sum.to_frame().join(tmp2); del tmp2 # merging series
# tmp = np.repeat(events_df_bp_sum.loc[(slice(None), 'Clinical Guidelines'), 'events'], len(polnames)) # subsetting values corresponding to no treatment in multiindex and repeating them the number of policies
# print(((events_df_bp_sum.events-tmp.to_numpy())/events_df_bp_sum.wt*per_capita).round(2).to_numpy()) # printing events saved per_capita number of patients, compared to the clinical guidelines
# print(((events_df_bp_sum.events-tmp.to_numpy())/tmp.to_numpy()).round(4).to_numpy()) # printing percentage change in events saved over no treatment, compared to the clinical guidelines
#
# # ##### Plotting results
# # os.chdir(fig_dir)
# # events_df_bp = events_df_bp[events_df_bp.bp_cat!=bp_cat_labels[0]] # removing normal BP category
# # events_df_bp['bp_cat'] = events_df_bp.bp_cat.cat.set_categories(bp_cat_labels[1:]) # adding sorted categories
# # qalys_events(events_df_bp, events=True)
#
# # Making summary of price of interpretability and optimality gap
# ## J_notrt = [np.dot(alpha.flatten(), V.flatten()) for V in pt_sim['V_notrt']] # calculating total expected discounted reward
#
# ## Data frame of expected total price of interpretability
# tmp = [[pt_sim['J_opt'][i] - y for i, y in enumerate(pt_sim[x])] for x in ['J_opt', 'J_mopt', 'J_q_learn', 'J_mlearn', 'J_aha']] # calculating price of interpretability per policy and removing optimal policy value functions
# tmp1 = pd.DataFrame(tmp).T; tmp1.columns = polnames # creating data frame from list of lists
# pi_df = pd.concat([ptdata1.loc[ptdata1.id.isin(pt_sim['pt_id']), ['id', 'wt', 'bp_cat']].reset_index(drop=True), tmp1], axis=1); del tmp1 # creating data frames with ids, weights, and BP categories
# pi_df = pi_df.melt(id_vars=['id', 'wt', 'bp_cat'], var_name='policy', value_name='J') # melting data frame
# pi_df['J'] = pi_df['J'].multiply(pi_df['wt'], axis=0) # adjusting value functions with sampling weights
#
# ### Overall PI per policy
# tmp = pi_df.groupby(['policy'], observed=False).wt.sum() # adding up sampling weights
# pi_ovr = pi_df.copy() # generating working copy of pi_df
# pi_ovr['policy'] = pi_ovr['policy'].astype('category') # converting policy to category
# pi_ovr['policy'] = pi_ovr.policy.cat.set_categories(polnames) # adding sorted categories
# pi_ovr = pi_ovr.sort_values(['policy']) # sorting dataframe based on selected columns
# pi_ovr = pi_ovr.groupby(['policy'], observed=False).J.sum() # summary of total expected reward
# pi_ovr = pi_ovr.to_frame().join(tmp); del tmp # merging series
# print((pi_ovr.J/pi_ovr.wt*per_capita).round(2).to_numpy()) # printing PI per_capita number of patients
# print(((pi_ovr.J.iloc[-1]-pi_ovr.J)/pi_ovr.J.iloc[-1]).round(4).to_numpy()) # printing percentage change in PI, compared to the clinical guidelines
#
# #### PI per capita (by BP category and policy)
# tmp2 = pi_df.groupby(['bp_cat', 'policy'], observed=False).wt.sum() # adding up sampling weights
# pi_df_bp = pi_df.copy() # generating working copy of pi_df
# pi_df_bp['bp_cat'] = pi_df_bp['bp_cat'].astype('category') # converting BP categories to category
# pi_df_bp['bp_cat'] = pi_df_bp.bp_cat.cat.set_categories(bp_cat_labels) # adding sorted categories
# pi_df_bp['policy'] = pi_df_bp['policy'].astype('category') # converting policy to category
# pi_df_bp['policy'] = pi_df_bp.policy.cat.set_categories(polnames) # adding sorted categories
# pi_df_bp = pi_df_bp.sort_values(['bp_cat', 'policy']) # sorting dataframe based on selected columns
# pi_df_bp_sum = pi_df_bp.groupby(['bp_cat', 'policy'], observed=False).J.sum() # summary of J saved, compared to no treatment
# pi_df_bp_sum = pi_df_bp_sum.to_frame().join(tmp2); del tmp2 # merging series
# tmp = np.repeat(pi_df_bp_sum.loc[(slice(None), 'Clinical Guidelines'), 'J'], len(polnames)) # subsetting values corresponding to no treatment in multiindex and repeating them the number of policies
# print((pi_df_bp_sum.J/pi_df_bp_sum.wt*per_capita).round(2).to_numpy()) # printing PI per_capita number of patients
# print(((tmp.to_numpy()-pi_df_bp_sum.J)/tmp.to_numpy()).round(4).to_numpy()) # printing percentage change in J, compared to the clinical guidelines

# ------------------------
# Patient-level analysis
# ------------------------

# Plotting policies for selected patient profiles
# dist_id_select = [539, 472, 1116, 1660]
# dist_ind_select = np.searchsorted(pt_sim['pt_id'], dist_id_select)

# # Identifying characteristics of patient profiles
# profile_df = ptdata1.loc[ptdata1.id.isin(pt_sim['pt_id'])].round(0)
# profile_df[['id','bp_cat']]

## Keeping only patients of interest
# pt_sim = {k: [v[i] for i in dist_ind_select] for k, v in pt_sim.items()}

# Loading results for patient profiles
os.chdir(results_dir)
with open('Results for patient profiles with 100000 learning iterations.pkl', 'rb') as f:
    pt_sim = pk.load(f)

## Creating dataframe of policies
policy_df = pd.DataFrame()
s = alive; t = list(range(years))[-1] # selecting alive states and last year of the planning horizon
for i in range(len(pt_sim['pt_id'])):
    # Combining results into a single data frame
    tmp = pd.concat([

        # For single year plots by state
        pd.Series(np.repeat(pt_sim['pt_id'][i], len(alive)), name='profile'),
        pd.Series(np.arange(len(alive)), name='state_id'),
        pd.Series(['Healthy', 'HA', 'H-HA', 'Stroke', 'H-Stroke', 'H-Both'], name='state'),

        # Policies
        pd.Series(pt_sim['d_opt'][i][s, t], name='OP'),
        # pd.Series(pt_sim['d_q_learn'][i][s, t], name='QL'),
        pd.Series(pt_sim['d_mopt'][i][s, t], name='MP'),
        pd.Series(pt_sim['d_mlearn'][i][s, t], name='MQL'),
        # pd.Series(pt_sim['d_aha'][i][s, t], name='AHA')

    ],
        axis=1)
    policy_df = pd.concat([policy_df, tmp], axis=0, ignore_index=True)

# Melting dataframe
policy_df = pd.melt(policy_df, id_vars=['profile', 'state_id', 'state'], var_name='policy', value_name='action')

## Identifying number of medications
policy_df['meds'] = np.select([policy_df.action == x for x in range(len(meds))], meds)

# Identifying policies
all_labels = ['0 SD/0 HD', '0 SD/1 HD', '1 SD/0 HD',
              '0 SD/2 HD', '1 SD/1 HD', '0 SD/3 HD', '2 SD/0 HD', # '2 SD/0 HD' = '0 SD/3 HD'
              '1 SD/2 HD', '0 SD/4 HD', '2 SD/1 HD', '1 SD/3 HD', '3 SD/0 HD', # '2 SD/1 HD' = '0 SD/4 HD'  # '3 SD/0 HD' = '1 SD/3 HD'
              '0 SD/5 HD', '2 SD/2 HD', '1 SD/4 HD', '3 SD/1 HD', '2 SD/3 HD', '4 SD/0 HD', # '2 SD/2 HD' = '0 SD/5 HD'  # '3 SD/1 HD' = '1 SD/4 HD' # '4 SD/0 HD' = '2 SD/3 HD'
              '3 SD/2 HD', '4 SD/1 HD', '5 SD/0 HD']
policy_df['labels'] = np.select([policy_df.action == x for x in range(len(all_labels))], all_labels)

# Plotting policies
os.chdir(fig_dir)
plot_policies_state(policy_df)

# Calculating PI in patient profiles
polnames = ['Optimal Monotone', 'Q-learning', 'Monotone Q-learning', 'Clinical Guidelines'] # names of policies
tmp = [[pt_sim['J_opt'][i] - y for i, y in enumerate(pt_sim[x])] for x in ['J_mopt', 'J_q_learn', 'J_mlearn', 'J_aha']] # calculating price of interpretability per policy and removing optimal policy value functions
tmp1 = pd.DataFrame(tmp, index=polnames).T # creating data frame from list of lists

# # Calculating percentage difference in events
# [(round(pt_sim['e_mopt'][i][0,0],4)-round(x[0,0],4))/round(x[0,0],4)*100 for i,x in enumerate(pt_sim['e_opt'])]
# [(round(pt_sim['e_mlearn'][i][0,0],4)-round(x[0,0],4))/round(x[0,0],4)*100 for i,x in enumerate(pt_sim['e_opt'])]