import numpy as np
import time
from tvb.simulator.lab import *
from toolbox import epochingTool, PLV, PLI, AEC, paramSpace, FFTpeak  # Home-made code
import mne.filter
import scipy.signal
import pandas as pd
import os
from datetime import datetime
import shutil


## This is the big computation chunk
# For each subject explore parameter space and use best performance to run a simulation
now=datetime.now()
new_folder=now.strftime("%m_%d_%Y-%H_%M_%S")
os.mkdir("paramSpace_exp/"+new_folder)

## Subjects
subj_list=["subj"+str(i).zfill(2) for i in np.arange(2,4,1)]

## parameters to explore
# Gvalue=np.arange(0,40,1)
# speed=np.arange(1,30,1)
# #TEST
Gvalue=[10]
speed=[7,2]

# Prepare simulation parameters
simLength = 8.1*1000 # ms
samplingFreq = 1000 #Hz

m = models.Generic2dOscillator()
int = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-4]))) # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
mon = (monitors.Raw(),)

for subj in subj_list:
    print("Working with %s's structure" % subj)
    os.mkdir("paramSpace_exp/"+new_folder+"/"+subj)

    conn = connectivity.Connectivity.from_file("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\CTB_connx66_"+subj+".zip")
    conn.weights=conn.weights/1e7
    results_fft_peak=list()
    results_fc=list()

    for g in Gvalue:
        for s in speed:

            tic = time.time()
            print("Simulating for %i coupling factor and %i mm/ms" % (g, s))
            coup = coupling.Linear(a=np.array(g))
            conn.speed = np.array([s])
            # Run simulation
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=int, monitors=mon)
            sim.configure()
            output = sim.run(simulation_length=simLength)


            # Extract data: "output[a][b][:,0,:,0].T" where:
            # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
            raw_data = output[0][1][100:, 0, :, 0].T
            raw_time = output[0][0][100:]

            results_fft_peak.append((g,s,FFTpeak(raw_data, simLength)))


            newRow=[g,s]
            bands = [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
            for b in range(len(bands[0])):
                (lowcut, highcut) = bands[1][b]

                # Band-pass filtering
                filterSignals = mne.filter.filter_data(raw_data,samplingFreq,lowcut,highcut)

                # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
                efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals for %s" % subj)

                # Obtain Analytical signal
                efPhase=list()
                efEnvelope = list()
                for i in range(len(efSignals)):
                    analyticalSignal = scipy.signal.hilbert(efSignals[i])
                    # Get instantaneous phase and amplitude envelope by channel
                    efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                    efEnvelope.append(np.abs(analyticalSignal))

                # CONNECTIVITY MEASURES
                plv = PLV(efPhase)
                aec = AEC(efEnvelope)

                # Load empirical data to make simple comparisons
                plv_emp=np.loadtxt("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_"+subj+"\\"+bands[0][b]+"plv.txt")
                aec_emp=np.loadtxt("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_"+subj+"\\"+bands[0][b]+"corramp.txt")

                # Comparisons
                t1 = np.zeros(shape=(2, 2145))
                t1[0, :] = plv[np.triu_indices(66, 1)]
                t1[1, :] = plv_emp[np.triu_indices(66, 1)]
                plv_r = np.corrcoef(t1)[0, 1]
                newRow.append(plv_r)

                t3 = np.zeros(shape=(2, 2145))
                t3[0, :] = aec[np.triu_indices(66, 1)]
                t3[1, :] = aec_emp[np.triu_indices(66, 1)]
                aec_r = np.corrcoef(t3)[0, 1]
                newRow.append(aec_r)

            results_fc.append(newRow)

            print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic,))


    ############ Working on FFT peak results
    df1=pd.DataFrame(results_fft_peak, columns=["G", "speed", "peak"])
    df1.to_csv("paramSpace_exp/"+ new_folder + "/"+subj+"/FFTpeaks_2d-8s_"+subj+".csv", index=False)
    paramSpace(df1, title="FFT peaks 2d"+subj, folder="paramSpace_exp/"+ new_folder + "/"+subj, auto_open=False)


    ######## Working on FC results
    df = pd.DataFrame(results_fc, columns=["G", "speed", "plvD_r", "aecD_r", "plvT_r", "aecT_r","plvA_r", "aecA_r",
                                           "plvB_r","aecB_r", "plvG_r", "aecG_r"])

    df.to_csv("paramSpace_exp/"+ new_folder + "/"+subj+"/FC_2d_8s_"+subj+".csv", index=False)

    dfPLV = df[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]]
    dfPLV.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
    paramSpace(dfPLV, 0.5, "2d_PLV_"+subj, folder="paramSpace_exp/"+ new_folder + "/"+subj, auto_open=False)

    dfAEC = df[["G", "speed", "aecD_r", "aecT_r", "aecA_r", "aecB_r", "aecG_r"]]
    dfAEC.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
    paramSpace(dfAEC, 0.5, "2d_AEC_"+subj, folder="paramSpace_exp/"+ new_folder + "/"+subj, auto_open=False)


    ## Looking for best values
    df["average"]=df[["plvD_r", "aecD_r", "plvT_r", "aecT_r","plvA_r", "aecA_r","plvB_r","aecB_r", "plvG_r", "aecG_r"]].mean(axis=1)
    df=df.sort_values(by=["average"],ascending=False, ignore_index=True)
    best_g=df["G"][0]
    best_s=df["speed"][0]

    ##################################################
    # Simulate again to gather comparable data
    subj_n=subj+subj[4:]
    os.mkdir("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_"+subj_n)
    shutil.copy("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_"+subj+"\\weights.txt",
                "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subj_n)

    print("Simulating %s with BEST PARAMETERS" % subj)

    tic = time.time()
    print("Simulating for %i coupling factor and %i mm/ms" % (best_g, best_s))
    coup = coupling.Linear(a=np.array(best_g))
    conn.speed = np.array([best_s])
    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=int, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    raw_data = output[0][1][100:, 0, :, 0].T
    raw_time = output[0][0][100:]

    bands = [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"],
             [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
    for b in range(len(bands[0])):
        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = mne.filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals for %s" % subj)

        # Obtain Analytical signal
        efPhase = list()
        efEnvelope = list()
        for i in range(len(efSignals)):
            analyticalSignal = scipy.signal.hilbert(efSignals[i])
            # Get instantaneous phase and amplitude envelope by channel
            efPhase.append(np.unwrap(np.angle(analyticalSignal)))
            efEnvelope.append(np.abs(analyticalSignal))

        # CONNECTIVITY MEASURES
        ## PLV
        plv = PLV(efPhase)
        fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subj_n+"\\"+bands[0][b]+"plv.txt"
        np.savetxt(fname, plv)

        ## PLI
        pli = PLI(efPhase)
        fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subj_n+"\\"+bands[0][b]+"pli.txt"
        np.savetxt(fname, pli)

        ## AEC
        aec = AEC(efEnvelope)
        fname="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subj_n+"\\"+bands[0][b]+"corramp.txt"
        np.savetxt(fname, aec)

del g, s, i, b, aec, aec_emp, aec_r, plv, plv_emp, plv_r, pli,
del analyticalSignal, efPhase, efSignals, efEnvelope, filterSignals, highcut, lowcut
del newRow, t1, t3, tic
del sim, coup, conn, m, int, models, samplingFreq, simLength, mon, output, raw_time, raw_data



################################################
# Measuring intersubject structural correlations in real data.
interW = np.zeros(shape=(9,9))

print("Structural correlations BETWEEN subjects", end="")
for s1 in range(2,11):
   print(".", end="")
   for s2 in range(2,11):
      s1dir="CTB_data\\output\\FC_subj"+str(s1)+"\\"
      s2dir="CTB_data\\output\\FC_subj"+str(s2)+"\\"

      weights1 = np.loadtxt(s1dir + "weights.txt")
      weights2 = np.loadtxt(s2dir + "weights.txt")

      t = np.zeros(shape=(2, 2145))
      t[0, :] = weights1[np.triu_indices(66, 1)]
      t[1, :] = weights2[np.triu_indices(66, 1)]
      interW[s1-2, s2-2] = np.corrcoef(t)[0, 1]

print(".")
W=pd.DataFrame(interW)
W.to_csv("interW.csv")
del t, weights1, weights2, s1dir, s2dir, s1, s2


# Getting number of subjects (could vary due to simulated new subjects)
files=os.listdir("CTB_data/output")[9:]
n=len(files)
names=[f[3:] for f in files]

# Checking correlations between subjects
# This is resting state FC, measures between subject should correlate to some extent
interAEC = np.zeros(shape=(len(bands[0]),n,n))
interPLV = np.zeros(shape=(len(bands[0]),n,n))
print("Calculating FC and SC correlations BETWEEN subjects", end="")
for i1, s1 in enumerate(names):
    print(".", end="")
    for i2, s2 in enumerate(names):
      s1dir="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_"+s1+"\\"
      s2dir="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_"+s2+"\\"

      for b in range(len(bands[0])):
         s1bdir = s1dir + bands[0][b]
         AEC1 = np.loadtxt(s1bdir + "corramp.txt")
         PLV1 = np.loadtxt(s1bdir + "plv.txt")

         s2bdir = s2dir + bands[0][b]
         AEC2 = np.loadtxt(s2bdir + "corramp.txt")
         PLV2 = np.loadtxt(s2bdir + "plv.txt")

         t1 = np.zeros(shape=(2, 2145))
         t1[0, :] = AEC1[np.triu_indices(66, 1)]
         t1[1, :] = AEC2[np.triu_indices(66, 1)]
         interAEC[b][i1, i2] = np.corrcoef(t1)[0, 1]

         t2 = np.zeros(shape=(2, 2145))
         t2[0, :] = PLV1[np.triu_indices(66, 1)]
         t2[1, :] = PLV2[np.triu_indices(66, 1)]
         interPLV[b][i1, i2] = np.corrcoef(t2)[0, 1]


for i, band in enumerate(bands[0]):
    i_plv=pd.DataFrame(interPLV[i], columns=names)
    i_plv.to_csv("interPLV_band"+band+".csv")
    i_aec=pd.DataFrame(interAEC[i], columns=names)
    i_aec.to_csv("interAEC_band"+band+".csv")
print(" %0.3f seconds.\n" % (time.time() - tic,))
del AEC1, AEC2, PLV1, PLV2, s1dir, s1bdir, s1, s2bdir, s2dir, s2, t1, t2, b, t3, weights2, weights1


# Checking structural - functional correlations within subjects
sfAEC = np.zeros(shape=(n, len(bands[0])))
sfPLV = np.zeros(shape=(n, len(bands[0])))
print("Calculating structural - functional correlations WITHIN subjects (PLV and AEC, independently)", end="")
for i, n in enumerate(names):
   print(".", end="")
   sdir="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_"+n+"\\"
   for b in range(len(bands)):
      AEC = np.loadtxt(sdir+bands[0][b]+"corramp.txt")
      PLV = np.loadtxt(sdir+bands[0][b]+"plv.txt")

      weights=np.loadtxt(sdir+"weights.txt")

      c1=np.zeros(shape=(2, 2145))
      c1[0,:]=AEC[np.triu_indices(66, 1)]
      c1[1,:]=weights[np.triu_indices(66, 1)]

      c2=np.zeros(shape=(2, 2145))
      c2[0,:]=PLV[np.triu_indices(66, 1)]
      c2[1,:]=weights[np.triu_indices(66, 1)]

      # Correlation between matrices
      sfAEC[i,b]=np.corrcoef(c1)[0,1]
      sfPLV[i,b]=np.corrcoef(c2)[0,1]

for i, band in enumerate(bands[0]):
    i_plv=pd.DataFrame(interPLV[i], columns=names)
    i_plv.to_csv("sfPLV"+band+".csv")
    i_aec=pd.DataFrame(interAEC[i], columns=names)
    i_aec.to_csv("sfAEC"+band+".csv")

print(" %0.3f seconds.\n" % (time.time() - tic,))

del s, b, sdir, AEC, PLV, weights, c1, c2
