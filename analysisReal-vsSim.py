import numpy as np
import scipy.stats
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio

names=np.loadtxt("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj2\\centres.txt", dtype="str")
names=names[:,0]

# Checking correlations within subjects between PLV and AEC
bands=["1-delta","2-theta","3-alfa","4-beta","5-gamma"]
intraR=np.zeros(shape=(9,5))
for s in range(9):
   sdir="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj"+str(s+2)+"\\"
   for b in range(len(bands)):
      sbdir=sdir+bands[b]
      AEC = np.loadtxt(sbdir+"corramp.txt")
      PLV = np.loadtxt(sbdir+"plv.txt")

      FC=np.zeros(shape=(2,2145))
      FC[0,:]=AEC[np.triu_indices(66,1)]
      FC[1,:]=PLV[np.triu_indices(66,1)]

      # Correlation between matrices
      intraR[s,b]=np.corrcoef(FC)[0,1]

np.average(intraR)


# Checking correlations between subjects
# This is resting state FC, measures between subject should correlate to some extent
interAEC = np.zeros(shape=(5,9,9))
interPLV = np.zeros(shape=(5,9,9))
for s1 in range(9):
   for s2 in range(9):
      s1dir="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj"+str(s1+2)+"\\"
      s2dir="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj"+str(s2+2)+"\\"

      for b in range(len(bands)):
         s1bdir = s1dir + bands[b]
         AEC1 = np.loadtxt(s1bdir + "corramp.txt")
         PLV1 = np.loadtxt(s1bdir + "plv.txt")

         s2bdir = s2dir + bands[b]
         AEC2 = np.loadtxt(s2bdir + "corramp.txt")
         PLV2 = np.loadtxt(s2bdir + "plv.txt")

         t1 = np.zeros(shape=(2, 2145))
         t1[0, :] = AEC1[np.triu_indices(66, 1)]
         t1[1, :] = AEC2[np.triu_indices(66, 1)]
         interAEC[b][s1, s2] = np.corrcoef(t1)[0, 1]

         t2 = np.zeros(shape=(2, 2145))
         t2[0, :] = PLV1[np.triu_indices(66, 1)]
         t2[1, :] = PLV2[np.triu_indices(66, 1)]
         interPLV[b][s1, s2] = np.corrcoef(t2)[0, 1]

np.average(interAEC[2]),np.average(interAEC[3]),np.average(interAEC[4]),np.average(interAEC[0]),np.average(interAEC[1])
np.average(interPLV[2]),np.average(interPLV[3]),np.average(interPLV[4]),np.average(interPLV[0]),np.average(interPLV[1])

_, p = scipy.stats.ttest_1samp(interAEC[2][0], 0.0)


# to plot just DMN
# indexes 2,8,23,25,26; 35,41,56,58,59
indexes=np.array([1,7,22,24,25,34,40,55,57,58])
DMNnames=names[indexes]
DMN=matrix[indexes,:][:,indexes]
fig = go.Figure(data=go.Heatmap(z=a, x=names, y=names, colorscale='Viridis'))
fig.update_layout(title='Phase Locking Value')
pio.write_html(fig, file="figures/PLV.html", auto_open=True)


y = np.arange(35).reshape(5,7)

y[1:5:2,::3]

y[np.array([0,2,4]), np.array([0,1,2])]

a=[[12,2,3],
   [2,4,5],
   [1,2,1]]

b=np.triu(a,1)