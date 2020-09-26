import numpy as np
import time
from tvb.simulator.lab import *
from toolbox import epochingTool, PLV, PLI, AEC, paramSpace, FFTpeak # Home-made code
import mne.filter
import scipy.signal
import pandas as pd

# Prepare simulation parameters
simLength = 8.5*1000 # ms
samplingFreq = 1024 #Hz

jrm = models.JansenRit(mu=np.array(0.09), p_max=np.array(0.15), p_min=np.array(0.03), v0=np.array(6.0), r=np.array(0.5))
phi_n_scaling = (jrm.a * jrm.A * (jrm.p_max - jrm.p_min) * 0.5) ** 2 / 2.
sigma = np.zeros(6)
sigma[3] = phi_n_scaling
int = integrators.HeunStochastic(dt=1000 / samplingFreq, noise=noise.Additive(
            nsig=sigma))  # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)

conn = connectivity.Connectivity.from_file("/CTB_data/output/CTB_connx66_subj04.zip")
conn.weights = conn.weights/1e7

mon = (monitors.Raw(),)

Gs = np.arange(0.5,90,3)
speed = np.arange(0.5,20,2)
results_fft_peak=list()
results_fc=list()

for g in Gs:
    for s in speed:

        coup = coupling.SigmoidalJansenRit(a=np.array(g))
        conn.speed = np.array([s])
        tic = time.time()
        print("Simulating for Coupling factor = %i and speed = %i " % (g, s))

        # Run simulation
        sim = simulator.Simulator(model=jrm, connectivity=conn, coupling=coup, integrator=int, monitors=mon)
        sim.configure()
        output = sim.run(simulation_length=simLength)

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
        raw_data = output[0][1][500:, 0, :, 0].T
        raw_time = output[0][0][500:]

        results_fft_peak.append((g,s,FFTpeak(raw_data, simLength)))


        newRow=[g,s]
        bands = [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
        for b in range(len(bands[0])):
            (lowcut, highcut) = bands[1][b]

            # Band-pass filtering
            filterSignals = mne.filter.filter_data(raw_data,samplingFreq,lowcut,highcut)

            # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
            efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

            # Obtain Analytical signal
            efPhase=list()
            efEnvelope = list()
            for i in range(len(efSignals)):
                analyticalSignal = scipy.signal.hilbert(efSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                efEnvelope.append(np.abs(analyticalSignal))

            # Check point
            # from toolbox import timeseriesPlot, plotConversions
            # regionLabels = conn.region_labels
            # timeseriesPlot(raw_data, raw_time, regionLabels)
            # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0], regionLabels, 1, raw_time)


            # CONNECTIVITY MEASURES
            ## PLV
            plv = PLV(efPhase)
            # fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"plv.txt"
            # np.savetxt(fname, plv)

            # ## PLI
            pli = PLI(efPhase)
            # fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"pli"
            # np.savetxt(fname, pli)

            ## AEC
            aec = AEC(efEnvelope)
            # fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"corramp.txt"
            # np.savetxt(fname, aec)

            # Load empirical data to make simple comparisons
            plv_emp=np.loadtxt("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj04\\"+bands[0][b]+"plv.txt")
            pli_emp=np.loadtxt("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj04\\"+bands[0][b]+"pli.txt")
            aec_emp=np.loadtxt("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj04\\"+bands[0][b]+"corramp.txt")

            # Comparisons
            t1 = np.zeros(shape=(2, 2145))
            t1[0, :] = plv[np.triu_indices(66, 1)]
            t1[1, :] = plv_emp[np.triu_indices(66, 1)]
            plv_r = np.corrcoef(t1)[0, 1]
            newRow.append(plv_r)

            t2 = np.zeros(shape=(2, 2145))
            t2[0, :] = pli[np.triu_indices(66, 1)]
            t2[1, :] = pli_emp[np.triu_indices(66, 1)]
            pli_r = np.corrcoef(t2)[0, 1]
            newRow.append(pli_r)

            t3 = np.zeros(shape=(2, 2145))
            t3[0, :] = aec[np.triu_indices(66, 1)]
            t3[1, :] = aec_emp[np.triu_indices(66, 1)]
            aec_r = np.corrcoef(t3)[0, 1]
            newRow.append(aec_r)

        results_fc.append(newRow)

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic,))


# Working on FFT peak results
df1 = pd.DataFrame(results_fft_peak, columns=["G", "speed", "peak"])

df1.to_csv("paramSpace_FFTpeaks-JR-subj4-8s-conn.csv", index=False)

# Load previously gathered data
# df1=pd.read_csv("expResults/paramSpace_FFTpeaks-2d-subj4-8s-sim.csv")
# df1a=df1[df1.G<33]
paramSpace(df1, title="JR")

# Working on FC results
df=pd.DataFrame(results_fc, columns=["G", "speed", "plvD_r", "pliD_r", "aecD_r", "plvT_r","pliT_r", "aecT_r","plvA_r",
                                  "pliA_r","aecA_r", "plvB_r","pliB_r","aecB_r", "plvG_r","pliG_r", "aecG_r"])

df.to_csv("paramSpace-JR-subj4-8s-conn.csv", index=False)


dfPLV = df[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]]
dfPLV.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
dfPLI = df[["G", "speed", "pliD_r", "pliT_r", "pliA_r", "pliB_r", "pliG_r"]]
dfPLI.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
dfAEC = df[["G", "speed", "aecD_r", "aecT_r", "aecA_r", "aecB_r", "aecG_r"]]
dfAEC.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]

del g, s, i, b, aec, aec_emp, aec_r, plv, plv_emp, plv_r, pli, pli_emp, pli_r
del analyticalSignal, bands, efPhase, efSignals, efEnvelope, filterSignals, highcut, lowcut
del newRow, phi_n_scaling, t1, t2, t3, tic, sigma
del sim, coup, conn, jrm, int, models, samplingFreq, simLength, mon, output, raw_time, raw_data


paramSpace(dfPLI, 0.5, "JR_PLI")
paramSpace(dfAEC, 0.5, "JR_AEC")
paramSpace(dfPLV, 0.5, "JR_PLV")

# df.to_csv("loop0-2m.csv", index=False)
# b=df.sort_values(by=["noise", "G"])
# fil=df[["G", "noise", "plv_avg","aec_avg","Tavg"]]
# fil=fil.sort_values(by=["noise", "G"])

# for i in range(len(plv)):
#     plv[i][i]=np.average(plv)