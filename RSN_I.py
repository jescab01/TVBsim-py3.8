import numpy as np
import time
from tvb.simulator.lab import *
from toolbox import epochingTool,bandpassFIRfilter,PLV, PLI, AEC  # Home-made code

# Prepare simulation components
simLength = 5*1000 # ms
m = models.ReducedWongWang()
#model = models.Generic2dOscillator(a=np.array(0.0))
conn = connectivity.Connectivity.from_file("connectivity_66.zip")
coup = coupling.Linear(a=np.array(6e-2))
int = integrators.HeunStochastic(dt=0.5, noise=noise.Additive(nsig=np.array([5e-4])))  # 2000 Hz
RAWmonitor = monitors.Raw()
mon = (RAWmonitor, monitors.TemporalAverage(period=5.0))

# Run simulation
tic = time.time()
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=int, monitors=mon)
sim.configure()
output = sim.run(simulation_length=simLength)
print("simulation required %0.3f seconds.\n" % (time.time() - tic,))

# Extract data: "output[a][b][:,0,:,0].T"
# where: a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
raw_data = output[0][1][200:, 0, :, 0].T
raw_time = output[0][0][200:]
regionLabels = conn.region_labels

# Epoch timeseries into 4 seconds windows
epochedData=epochingTool(raw_data, 1, 2000)


bands=[["1-delta","2-theta","3-alfa","4-beta","5-gamma"],[(2,4),(4,8),(8,12),(12,30),(30,45)]]
for b in range(len(bands[0])):
    (lowcut,highcut)=bands[1][b]

    # Band-pass filtering
    # Famous window functions are: bartlett, hann, hamming, blackmann.
    filterSignals = bandpassFIRfilter(epochedData, lowcut, highcut, "hann", 2000)

    # CONNECTIVITY MEASURES
    ## PLV
    plv = PLV(filterSignals)
    fname="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj1\\"+bands[0][b]+"plv.txt"
    np.savetxt(fname, plv)

    ## PLI
    pli = PLI(filterSignals)
    fname="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj1\\"+bands[0][b]+"pli"
    np.savetxt(fname, pli)

    ## AEC
    aec = AEC(filterSignals)
    fname="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj1\\"+bands[0][b]+"corramp.txt"
    np.savetxt(fname, aec)






# x=np.reshape(plv,65*65)
# y=np.reshape(pli,65*65)     # No correlacionan en absoluto. ¿Es posible?
# np.corrcoef(x,y)
