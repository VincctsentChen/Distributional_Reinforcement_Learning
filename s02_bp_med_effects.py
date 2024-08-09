# ======================================================
# Calculating reductions in BP and risk due to treatment
# ======================================================

# Loading modules
import numpy as np
import itertools as it  # recursive operations

# ---------------------------------------------------------------
# Using parameters in Sundstrom et al. 2014 and 2015 (base case)
# ---------------------------------------------------------------

# Drug combination function (sorted in non-decreasing order of actions -
# sorted by absolute risk reduction with ties broken with their estimated disutility)
def med_effects(hf_red_frac, sbp_drop_std, sbp_drop_hf, dbp_drop_std, dbp_drop_hf, rel_risk_chd_std, rel_risk_chd_hf,
                rel_risk_stroke_std, rel_risk_stroke_hf, disut_std, disut_hf, numtrt, order):

    # Storing reductions and disutilities due to medications
    sbp_reduction = np.empty(numtrt); dbp_reduction = np.empty(numtrt); rel_risk_chd = np.empty(numtrt)
    rel_risk_stroke = np.empty(numtrt); trtharm = np.empty(numtrt); meds = np.empty(numtrt)

    # we have 21 possible treatments in total (21 actions)
    for trt in range(numtrt):
        if trt == 0:  # no treatment
            sbp_reduction[trt] = 0
            dbp_reduction[trt] = 0
            rel_risk_chd[trt] = 1
            rel_risk_stroke[trt] = 1
            trtharm[trt] = 0
            meds[trt] = 0
        elif trt == 1:  # 1 half
            sbp_reduction[trt] = sbp_drop_hf
            dbp_reduction[trt] = dbp_drop_hf
            rel_risk_chd[trt] = rel_risk_chd_hf
            rel_risk_stroke[trt] = rel_risk_stroke_hf
            trtharm[trt] = disut_hf
            meds[trt] = hf_red_frac
        elif trt == 2:  # 1 standard
            sbp_reduction[trt] = sbp_drop_std
            dbp_reduction[trt] = dbp_drop_std
            rel_risk_chd[trt] = rel_risk_chd_std
            rel_risk_stroke[trt] = rel_risk_stroke_std
            trtharm[trt] = disut_std
            meds[trt] = 1
        elif trt == 3:  # 2 half
            sbp_reduction[trt] = sbp_drop_hf*2
            dbp_reduction[trt] = dbp_drop_hf*2
            rel_risk_chd[trt] = rel_risk_chd_hf**2
            rel_risk_stroke[trt] = rel_risk_stroke_hf**2
            trtharm[trt] = disut_hf*2
            meds[trt] = hf_red_frac*2
        elif trt == 4:  # 1 standard, 1 half
            sbp_reduction[trt] = sbp_drop_std + sbp_drop_hf
            dbp_reduction[trt] = dbp_drop_std + dbp_drop_hf
            rel_risk_chd[trt] = rel_risk_chd_std*rel_risk_chd_hf
            rel_risk_stroke[trt] = rel_risk_stroke_std*rel_risk_stroke_hf
            trtharm[trt] = disut_std + disut_hf
            meds[trt] = 1 + hf_red_frac
        elif trt == 5:  # 3 half
            sbp_reduction[trt] = sbp_drop_hf*3
            dbp_reduction[trt] = dbp_drop_hf*3
            rel_risk_chd[trt] = rel_risk_chd_hf**3
            rel_risk_stroke[trt] = rel_risk_stroke_hf**3
            trtharm[trt] = disut_hf*3
            meds[trt] = hf_red_frac*3
        elif trt == 6:  # 2 standard
            sbp_reduction[trt] = sbp_drop_std*2
            dbp_reduction[trt] = dbp_drop_std*2
            rel_risk_chd[trt] = rel_risk_chd_std**2
            rel_risk_stroke[trt] = rel_risk_stroke_std**2
            trtharm[trt] = disut_std*2
            meds[trt] = 2
        elif trt == 7:  # 1 standard, 2 half
            sbp_reduction[trt] = sbp_drop_std + sbp_drop_hf*2
            dbp_reduction[trt] = dbp_drop_std + dbp_drop_hf*2
            rel_risk_chd[trt] = rel_risk_chd_std*(rel_risk_chd_hf**2)
            rel_risk_stroke[trt] = rel_risk_stroke_std*(rel_risk_stroke_hf**2)
            trtharm[trt] = disut_std + disut_hf*2
            meds[trt] = 1 + hf_red_frac*2
        elif trt == 8:  # 4 half
            sbp_reduction[trt] = sbp_drop_hf*4
            dbp_reduction[trt] = dbp_drop_hf*4
            rel_risk_chd[trt] = rel_risk_chd_hf**4
            rel_risk_stroke[trt] = rel_risk_stroke_hf**4
            trtharm[trt] = disut_hf*4
            meds[trt] = hf_red_frac*4
        elif trt == 9:  # 2 standard, 1 half
            sbp_reduction[trt] = sbp_drop_std*2 + sbp_drop_hf
            dbp_reduction[trt] = dbp_drop_std*2 + dbp_drop_hf
            rel_risk_chd[trt] = (rel_risk_chd_std**2)*rel_risk_chd_hf
            rel_risk_stroke[trt] = (rel_risk_stroke_std**2)*rel_risk_stroke_hf
            trtharm[trt] = disut_std*2 + disut_hf
            meds[trt] = 2 + hf_red_frac
        elif trt == 10:  # 1 standard, 3 half
            sbp_reduction[trt] = sbp_drop_std + sbp_drop_hf*3
            dbp_reduction[trt] = dbp_drop_std + dbp_drop_hf*3
            rel_risk_chd[trt] = rel_risk_chd_std*(rel_risk_chd_hf**3)
            rel_risk_stroke[trt] = rel_risk_stroke_std*(rel_risk_stroke_hf**3)
            trtharm[trt] = disut_std + disut_hf*3
            meds[trt] = 1 + hf_red_frac*3
        elif trt == 11:  # 3 standard
            sbp_reduction[trt] = sbp_drop_std*3
            dbp_reduction[trt] = dbp_drop_std*3
            rel_risk_chd[trt] = rel_risk_chd_std**3
            rel_risk_stroke[trt] = rel_risk_stroke_std**3
            trtharm[trt] = disut_std*3
            meds[trt] = 3
        elif trt == 12:  # 5 half
            sbp_reduction[trt] = sbp_drop_hf*5
            dbp_reduction[trt] = dbp_drop_hf*5
            rel_risk_chd[trt] = rel_risk_chd_hf**5
            rel_risk_stroke[trt] = rel_risk_stroke_hf**5
            trtharm[trt] = disut_hf*5
            meds[trt] = hf_red_frac*5
        elif trt == 13:  # 2 standard, 2 half
            sbp_reduction[trt] = sbp_drop_std*2 + sbp_drop_hf*2
            dbp_reduction[trt] = dbp_drop_std*2 + dbp_drop_hf*2
            rel_risk_chd[trt] = (rel_risk_chd_std**2)*(rel_risk_chd_hf**2)
            rel_risk_stroke[trt] = (rel_risk_stroke_std**2)*(rel_risk_stroke_hf**2)
            trtharm[trt] = disut_std*2 + disut_hf*2
            meds[trt] = 2 + hf_red_frac*2
        elif trt == 14:  # 1 standard, 4 half
            sbp_reduction[trt] = sbp_drop_std + sbp_drop_hf*4
            dbp_reduction[trt] = dbp_drop_std + dbp_drop_hf*4
            rel_risk_chd[trt] = rel_risk_chd_std*(rel_risk_chd_hf**4)
            rel_risk_stroke[trt] = rel_risk_stroke_std*(rel_risk_stroke_hf**4)
            trtharm[trt] = disut_std + disut_hf*4
            meds[trt] = 1 + hf_red_frac*4
        elif trt == 15:  # 3 standard, 1 half
            sbp_reduction[trt] = sbp_drop_std*3 + sbp_drop_hf
            dbp_reduction[trt] = dbp_drop_std*3 + dbp_drop_hf
            rel_risk_chd[trt] = (rel_risk_chd_std**3)*rel_risk_chd_hf
            rel_risk_stroke[trt] = (rel_risk_stroke_std**3)*rel_risk_stroke_hf
            trtharm[trt] = disut_std*3 + disut_hf
            meds[trt] = 3 + hf_red_frac
        elif trt == 16:  # 2 standard, 3 half
            sbp_reduction[trt] = sbp_drop_std*2 + sbp_drop_hf*3
            dbp_reduction[trt] = dbp_drop_std*2 + dbp_drop_hf*3
            rel_risk_chd[trt] = (rel_risk_chd_std**2)*(rel_risk_chd_hf**3)
            rel_risk_stroke[trt] = (rel_risk_stroke_std**2)*(rel_risk_stroke_hf**3)
            trtharm[trt] = disut_std*2 + disut_hf*3
            meds[trt] = 2 + hf_red_frac*3
        elif trt == 17:  # 4 standard
            sbp_reduction[trt] = sbp_drop_std*4
            dbp_reduction[trt] = dbp_drop_std*4
            rel_risk_chd[trt] = rel_risk_chd_std**4
            rel_risk_stroke[trt] = rel_risk_stroke_std**4
            trtharm[trt] = disut_std*4
            meds[trt] = 4
        elif trt == 18:  # 3 standard, 2 half
            sbp_reduction[trt] = sbp_drop_std*3 + sbp_drop_hf*2
            dbp_reduction[trt] = dbp_drop_std*3 + dbp_drop_hf*2
            rel_risk_chd[trt] = (rel_risk_chd_std**3)*(rel_risk_chd_hf**2)
            rel_risk_stroke[trt] = (rel_risk_stroke_std**3)*(rel_risk_stroke_hf**2)
            trtharm[trt] = disut_std*3 + disut_hf*2
            meds[trt] = 3 + hf_red_frac*2
        elif trt == 19:  # 4 standard, 1 half
            sbp_reduction[trt] = sbp_drop_std*4 + sbp_drop_hf
            dbp_reduction[trt] = dbp_drop_std*4 + dbp_drop_hf
            rel_risk_chd[trt] = (rel_risk_chd_std**4)*rel_risk_chd_hf
            rel_risk_stroke[trt] = (rel_risk_stroke_std**4)*rel_risk_stroke_hf
            trtharm[trt] = disut_std*4 + disut_hf
            meds[trt] = 4 + hf_red_frac
        elif trt == 20:  # 5 standard
            sbp_reduction[trt] = sbp_drop_std*5
            dbp_reduction[trt] = dbp_drop_std*5
            rel_risk_chd[trt] = rel_risk_chd_std**5
            rel_risk_stroke[trt] = rel_risk_stroke_std**5
            trtharm[trt] = disut_std*5
            meds[trt] = 5

        # print([trt, np.round(sbp_reduction[trt], 2), np.round(dbp_reduction[trt], 2),
        #        np.round(1-rel_risk_chd[trt], 2), np.round(1-rel_risk_stroke[trt], 2),
        #        np.round(trtharm[trt], 3), frac.Fraction(meds[trt]).limit_denominator(3)])

    if order == "nonincreasing": # sorting actions in non-increasing order
        sbp_reduction, dbp_reduction = np.array(list(reversed(sbp_reduction))), np.array(list(reversed(dbp_reduction)))
        rel_risk_chd, rel_risk_stroke = np.array(list(reversed(rel_risk_chd))), np.array(list(reversed(rel_risk_stroke)))
        trtharm, meds = np.array(list(reversed(trtharm))), np.array(list(reversed(meds)))

    return sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, trtharm, meds

