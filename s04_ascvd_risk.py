# =========================
# Calculating ASCVD risk
# ========================

# Loading modules
import math
import sys

# ASCVD risk calculator (Yadlowsky et al. 2018)
def arisk(event, sex, race, age, sbp, smk, tc, hdl, diab, trt, time):
    # ASCVD risk calculator (Yadlowsky et al. 2018)
    # inputs: type of risk to calculate (1=CHD, 2=stroke), sex (1=male, 0=female), race (1=white,0=black)
    # age, SBP, smoking status (1=smoker, 0=nonsmoker), total cholesterol, HDL, diabetes status (1=diabetic, 0=nondiabetic),
    # trt (1=BP reported is on treatment, 0=BP reported is untreated), time (1-year, 10-year risk).
    # outputs: likelihood of CHD or stroke in the next "time" years

    # Generating indicator of Black race from race column
    if race == 1:
        black = 0
    else:
        black = 1

    # Coefficients for male patients
    if sex == 1:
        intercept = -11.679980
        b_age = 0.064200
        b_black = 0.482835
        b_sbp2 = -0.000061
        b_sbp = 0.038950
        b_trt = 2.055533
        b_diab = 0.842209
        b_smk = 0.895589
        b_tc_hdl = 0.193307
        b_black_age = 0
        b_trt_sbp = -0.014207
        b_black_sbp = 0.011609
        b_black_trt = -0.119460
        b_age_sbp = 0.000025
        b_black_diab = -0.077214
        b_black_smk = -0.226771
        b_black_tc_hdl = -0.117749
        b_black_trt_sbp = 0.004190
        b_black_age_sbp = -0.000199

    # Coefficients for female patients
    else:
        intercept = -12.823110
        b_age = 0.106501
        b_black = 0.432440
        b_sbp2 = 0.000056
        b_sbp = 0.017666
        b_trt = 0.731678
        b_diab = 0.943970
        b_smk = 1.009790
        b_tc_hdl = 0.151318
        b_black_age = -0.008580
        b_trt_sbp = -0.003647
        b_black_sbp = 0.006208
        b_black_trt = 0.152968
        b_age_sbp = -0.000153
        b_black_diab = 0.115232
        b_black_smk = -0.092231
        b_black_tc_hdl = 0.070498
        b_black_trt_sbp = -0.000173
        b_black_age_sbp = -0.000094

    # Proportion of ascvd assumed to be CHD or stroke, respectively
    eventprop = [0.7, 0.3] # updated 1/23/2020 after email with Sussman

    # Calculating sum of terms
    betaX = intercept+b_age*age+b_black*black+b_sbp2*(sbp**2)+b_sbp*sbp+b_trt*trt+b_diab*diab+ \
            b_smk*smk+b_tc_hdl*(tc/hdl)+b_black_age*(black*age)+b_trt_sbp*(trt*sbp)+b_black_sbp*(black*sbp)+ \
            b_black_trt*(black*trt)+b_age_sbp*(age*sbp)+b_black_diab*(black*diab)+b_black_smk*(black*smk)+ \
            b_black_tc_hdl*(black*tc/hdl)+b_black_trt_sbp*(black*trt*sbp)+b_black_age_sbp*(black*age*sbp)

    # Estimating risk
    risk = eventprop[event]*(1/(1+(math.exp(-betaX))))

    # Estimating 1-year or 10-year risk for ASCVD events
    if time == 1:
        mult = 0.082 # this multiplier was obtained by solving the LP in the '1_year_ascvd_calibration.py' script
        a_risk = risk*mult
    elif time == 10:
        mult = 1
        a_risk = risk*mult
    else:
        sys.exit(str(time)+" is an improper time length for risk calculation")

    return a_risk