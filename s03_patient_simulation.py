# =======================================================
# Patient Simulation - Hypertension treatment case study
# =======================================================

# Loading modules
import numpy as np  # array operations
import time as tm  # timing code
from s04_ascvd_risk import arisk  # risk calculations
from s05_transition_probabilities import TP  # transition probability calculations
from s06_optimal_monotone_mdp import lp_mdp_dual, mip_mdp_dual_epochs, q_learning, Monotonic_Q_learning, aha_eval  # MDPs, Q-learning, and no treatment policy
from s08_aha_2017_guideline import aha_guideline # 2017 AHA's guidelines

# # Lines for debugging purposes (run hypertension_treatment_monotone_qlearning.py until Initializing parameters analyses section first) # pt_id = 2463 was selected for example calculations in appendix
#pt_id = 593; patientdata = ptdata[ptdata.id == pt_id] # testing function on a single patient
# Run 500 patients here. [539, 472, 1116, 1660]
#pt_id = 539; patientdata = ptdata[ptdata.id == pt_id] #run DRL for patient_id = 539:
#pt_id = 472; patientdata = ptdata[ptdata.id == pt_id] #run DRL for patient_id = 472:
#pt_id = 1116; patientdata = ptdata[ptdata.id == pt_id] #run DRL for patient_id = 1116:
pt_id = 1660; patientdata = ptdata[ptdata.id == pt_id] #run DRL for patient_id = 1660:


# Patient simulation function
def patient_sim(pt_id, patientdata, numhealth, healthy, dead, years, events, stroke_hist, ascvd_hist_scale, event_states,
                lifedata, mortality_rates, chddeathdata, strokedeathdata, alldeathdata, sbpmin, dbpmin,
                sbpmax, dbpmax, allmeds, trtharm, sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke,
                QoL, QoLterm, alpha, gamma, state_order, N, epsilon, targetrisk, targetsbp, targetdbp): #, J_mopt

    """"
    This function generates risk estimates and transition probabilities
    to determine treatment policies per patient
    """""

    # Assume that the patient has the same pre-treatment SBP/DBP no matter the health condition
    pretrtsbp = np.ones([numhealth, years])*np.array(patientdata.sbp)
    pretrtdbp = np.ones([numhealth, years])*np.array(patientdata.dbp)

    # Storing risk calculations
    ascvdrisk1 = np.empty((numhealth, years, events))  # 1-y CHD and stroke risk (for transition probabilities)
    periodrisk1 = np.empty((numhealth, years, events))  # 1-y risk after scaling

    ascvdrisk10 = np.empty((numhealth, years, events))  # 10-y CHD and stroke risk (for AHA's guidelines)
    periodrisk10 = np.empty((numhealth, years, events))  # 10-y risk after scaling

    for h in range(numhealth):  # each state (in order of rewards)
        for t in range(years):  # each age

            # Changing scaling factor based on age
            if patientdata.age.iloc[t] >= 60:
                ascvd_hist_scale_sim = ascvd_hist_scale
                ascvd_hist_scale_sim[stroke_hist, 1] = 2
            else:
                ascvd_hist_scale_sim = ascvd_hist_scale

            for k in range(events):  # each event type

                # 1-year ASCVD risk calculation (for transition probabilities)
                ascvdrisk1[h, t, k] = arisk(k, patientdata.sex.iloc[t], patientdata.race.iloc[t], patientdata.age.iloc[t],
                                            patientdata.sbp.iloc[t], patientdata.smk.iloc[t], patientdata.tc.iloc[t],
                                            patientdata.hdl.iloc[t], patientdata.diab.iloc[t], 0, 1)

                # 10-year ASCVD risk calculation (for AHA's guidelines)
                ascvdrisk10[h, t, k] = arisk(k, patientdata.sex.iloc[t], patientdata.race.iloc[t], patientdata.age.iloc[t],
                                             patientdata.sbp.iloc[t], patientdata.smk.iloc[t], patientdata.tc.iloc[t],
                                             patientdata.hdl.iloc[t], patientdata.diab.iloc[t], 0, 10)

                if ascvd_hist_scale_sim[h, k] > 1:
                    # Scaling odds of 1-year risks
                    periododds = ascvdrisk1[h, t, k]/(1-ascvdrisk1[h, t, k])
                    periododds = ascvd_hist_scale_sim[h, k]*periododds
                    periodrisk1[h, t, k] = periododds/(1+periododds)

                    # Scaling odds of 10-year risks
                    periododds = ascvdrisk10[h, t, k]/(1-ascvdrisk10[h, t, k])
                    periododds = ascvd_hist_scale_sim[h, k]*periododds
                    periodrisk10[h, t, k] = periododds/(1+periododds)

                elif ascvd_hist_scale_sim[h, k] == 0:  # set risk to 0
                    periodrisk1[h, t, k] = 0
                    periodrisk10[h, t, k] = 0
                else:  # no scale
                    periodrisk1[h, t, k] = ascvdrisk1[h, t, k]
                    periodrisk10[h, t, k] = ascvdrisk10[h, t, k]

    # life expectancy and death likelihood data index
    if patientdata.sex.iloc[0] == 1:  # male
        sexcol = 1  # column in deathdata corresponding to male
    else:
        sexcol = 2  # column in deathdata corresponding to female

    # Death rates
    chddeath = chddeathdata.iloc[list(np.where([i in list(patientdata.age) for i in list(chddeathdata.iloc[:, 0])])[0]), sexcol]
    strokedeath = strokedeathdata.iloc[list(np.where([i in list(patientdata.age) for i in list(strokedeathdata.iloc[:, 0])])[0]), sexcol]
    alldeath = alldeathdata.iloc[list(np.where([i in list(patientdata.age) for i in list(alldeathdata.iloc[:, 0])])[0]), sexcol]

    # Calculating transition probabilities
    P, feas = TP(periodrisk1, chddeath, strokedeath, alldeath, pretrtsbp, pretrtdbp, sbpmin, sbpmax, dbpmin, dbpmax,
                 sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke)

    # Sorting transition probabilities and feasibility indicators according to state ordering
    P = P[state_order, :, :, :]; P = P[:, state_order, :, :]; feas = feas[state_order, :, :]

    # Extracting list of infeasible actions per state and decision epoch
    infeasible = []  # stores index of infeasible actions (if infeasible=1 means the action is infeasible)
    feasible = []  # stores index of feasible actions (feaible =1 means action is feasible)
    for s in range(feas.shape[0]):
        tmp = []; tmp1 = []
        for t in range(feas.shape[1]):
            tmp.append(list(np.where(feas[s, t, :] == 0)[0]))
            tmp1.append(list(np.where(feas[s, t, :] == 1)[0]))
        infeasible.append(tmp); feasible.append(tmp1); del tmp, tmp1
    del feas

    # Calculating expected rewards
    r = np.empty((numhealth, years, len(allmeds))); r[:] = np.nan  # stores rewards
    for t in range(years):
        # QoL weights by age
        qol = None
        if 40 <= patientdata.age.iloc[t] <= 44:
            qol = QoL.get("40-44")
        elif 45 <= patientdata.age.iloc[t] <= 54:
            qol = QoL.get("45-54")
        elif 55 <= patientdata.age.iloc[t] <= 64:
            qol = QoL.get("55-64")
        elif 65 <= patientdata.age.iloc[t] <= 74:
            qol = QoL.get("65-74")
        elif 75 <= patientdata.age.iloc[t] <= 84:
            qol = QoL.get("75-84")
        qol = np.array(qol)[state_order]  # Ordering rewards

        # Subtracting treatment disutility
        for a in range(len(allmeds)):
            r[:, t, a] = [max(0, rw-trtharm[a]) for rw in qol]  # bounding rewards below by zero (assuming there is nothing worse than death)

    # Terminal conditions
    ## Healthy life expectancy
    healthy_lifexp = lifedata.iloc[np.where(patientdata.age.iloc[max(range(years))] == lifedata.iloc[:, 0])[0][0], sexcol]

    ## Mortality rates by gender
    if patientdata.sex.iloc[0] == 0:  # Male mortality rates
        SMR = mortality_rates.get("Males <2 CHD events")
    else:  # Female mortality rates
        SMR = mortality_rates.get("Females <2 CHD events")

    ## Calculating terminal rewards (terminal QALYs by age)
    rterm = None
    if 40 <= patientdata.age.iloc[max(range(years))] <= 44:
        rterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("40-44"))]
    elif 45 <= patientdata.age.iloc[max(range(years))] <= 54:
        rterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("45-54"))]
    elif 55 <= patientdata.age.iloc[max(range(years))] <= 64:
        rterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("55-64"))]
    elif 65 <= patientdata.age.iloc[max(range(years))] <= 74:
        rterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("65-74"))]
    elif 75 <= patientdata.age.iloc[max(range(years))] <= 84:
        rterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("75-84"))]
    rterm = np.array(rterm)[state_order] # Ordering terminal rewards

    # Ordering state distribution
    alpha1 = np.ones(alpha.shape)  # initial state distribution for LP
    alpha = alpha[state_order, :] # initial state distribution for MIP

    # # Determining optimal policies (using dual formulation)
    # V_opt, d_opt, occup, J_opt, e_opt = lp_mdp_dual(P, r, rterm, alpha1, gamma, infeasible, event_states)
    # d_opt[dead, :] = 0  # treating only on alive states
    #
    # ## Correcting total expected discounted reward with actual initial state distribution
    # J_opt = np.dot(alpha.flatten(), V_opt.flatten())
    #
    # # Generating monotone policy from optimal solution (enforcing monotonicity by increasing actions) to warm start MIPs
    # ws_mopt = np.empty((numhealth, years, len(allmeds))); ws_mopt[:] = np.nan
    # for h in range(numhealth):
    #     for t in range(years):
    #         for a in range(len(allmeds)):
    #             ws_mopt[h, t, a] = np.where(d_opt[h, t] == a, 1, 0)
    #         if h > 0:
    #             if np.argmax(ws_mopt[h, t, :]) < np.argmax(ws_mopt[(h-1), t, :]):
    #                 ws_mopt[h, t, int(np.argmax(ws_mopt[h, t, :]))] = 0
    #                 ws_mopt[h, t, int(np.argmax(ws_mopt[(h-1), t, :]))] = 1
    #
    # # Determining monotone policies in states and decision epochs (using dual formulation)
    # V_mopt, d_mopt, J_mopt, e_mopt = mip_mdp_dual_epochs(P, r, rterm, alpha, gamma, infeasible, event_states, J_opt, ws_mopt)
    # d_mopt[dead, :] = 0  # treating only on alive states
    #
    # # Calculating value functions and policies using traditional Q-learning
    # V_q_learn, d_q_learn, J_q_learn, e_q_learn = q_learning(P, r, rterm, alpha, gamma, N, epsilon, healthy, feasible, event_states)
    # d_q_learn[dead, :] = 0

    # Call to monotone Q-learning
    # Delete everything in monotonic q learning and just keep materials related to episode
    # use this method to generate action: a_now = np.random.choice(np.arange(b.shape[2]), p=b[s_now, t, :])   
    # use this code to select s_next: s_next = np.random.choice(np.arange(S), p=P[s_now, :, t, a_now]) # sampling next state
    # reward: r[s_now, t, a_now]
    # use this to indicae if the episode terminates: if (s_next==6 or s_next==7 or s_next==8 or s_next==9) or t == max(range(T)): then done = True; else done = False
    V_mlearn, d_mlearn, J_mlearn, e_mlearn = Monotonic_Q_learning(P, r, rterm, alpha, gamma, N, epsilon, healthy, infeasible, event_states, 100000)
    d_mlearn[dead, :] = 0
    # 

    # # Determining policy based on the 2017 AHA's guidelines
    # d_aha = aha_guideline(periodrisk10, pretrtsbp, pretrtdbp, targetrisk, targetsbp, targetdbp, sbpmin, dbpmin,
    #                       sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, healthy)
    # d_aha[dead, :] = 0  # treating only on alive states
    #
    # # Evaluating the 2017 AHA's guidelines
    # V_aha, J_aha, e_aha = aha_eval(d_aha.astype(int), P, r, rterm, alpha, gamma, event_states)

    # # Evaluating no treatment policy
    # V_notrt, e_notrt = notrt_eval(P, r, rterm, gamma, event_states)

    #print(V_mlearn, d_mlearn, J_mlearn, e_mlearn)
    #print("Patient " + str(pt_id) + " Doneaa" ,"aa",V_mlearn, d_mlearn, J_mlearn, e_mlearn,"aa")

    # Keeping track of progress
    print(tm.asctime(tm.localtime(tm.time()))[:-5], "Patient", pt_id, "Done")

    return (pt_id, # V_notrt, e_notrt, # periodrisk, P,
            # V_opt, d_opt, J_opt, e_opt,
            # V_mopt, d_mopt, J_mopt, e_mopt,
            # V_aha, d_aha, J_aha, e_aha,
            # V_q_learn, d_q_learn, J_q_learn, e_q_learn,
            V_mlearn, d_mlearn, J_mlearn, e_mlearn
            )