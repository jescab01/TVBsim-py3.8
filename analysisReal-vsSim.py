import numpy as np
import scipy.stats
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
import os
import time

# Time reference
tic = time.time()

# Getting number of subjects (could vary due to simulated new subjects)
files=os.listdir("CTB_data/output")
n=len(files)-9 # Total - number of SC .zips - number of non desired subjects to analize
del files

# Load names
names = np.loadtxt("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj02\\centres.txt", dtype="str")
names = names[:,0]

# Load bands
bands = ["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"]

# Checking correlations within subjects in FC (i.e. between PLV and AEC)
intraPLVvsAEC = np.zeros(shape=(n, len(bands)))
print("Calculating correlations between PLV and AEC (WITHIN subjects)", end="")
for s in range(n):
   print(".", end="")
   sdir="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj"+str(s)+"\\"
   for b in range(len(bands)):
      sbdir=sdir+bands[b]
      AEC = np.loadtxt(sbdir+"corramp.txt")
      PLV = np.loadtxt(sbdir+"plv.txt")

      FC=np.zeros(shape=(2,2145))
      FC[0,:]=AEC[np.triu_indices(66,1)]
      FC[1,:]=PLV[np.triu_indices(66,1)]

      # Correlation between matrices
      intraPLVvsAEC[s,b]=np.corrcoef(FC)[0,1]

print(" %0.3f seconds.\n" % (time.time() - tic,))

del AEC, PLV, FC, s, sdir, sbdir

# np.average(intraR)

# Checking correlations between subjects
# This is resting state FC, measures between subject should correlate to some extent
interAEC = np.zeros(shape=(len(bands),n,n))
interPLV = np.zeros(shape=(len(bands),n,n))
interW = np.zeros(shape=(n,n))
print("Calculating FC and SC correlations BETWEEN subjects (PLV and AEC, independently)", end="")
for s1 in range(n):
   print(".", end="")
   for s2 in range(n):
      s1dir="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj"+str(s1)+"\\"
      s2dir="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj"+str(s2)+"\\"

      weights1 = np.loadtxt(s1dir + "weights.txt")
      weights2 = np.loadtxt(s2dir + "weights.txt")

      t3 = np.zeros(shape=(2, 2145))
      t3[0, :] = weights1[np.triu_indices(66, 1)]
      t3[1, :] = weights2[np.triu_indices(66, 1)]
      interW[s1, s2] = np.corrcoef(t3)[0, 1]

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

print(" %0.3f seconds.\n" % (time.time() - tic,))

del AEC1, AEC2, PLV1, PLV2, s1dir, s1bdir, s1, s2bdir, s2dir, s2, t1, t2, b, t3, weights2, weights1

# np.average(interW,1) # All of empirical SC matrices correlate very well with each other. Best performance subj4.
# np.average(interAEC[2]),np.average(interAEC[3]),np.average(interAEC[4]),np.average(interAEC[0]),np.average(interAEC[1])
# np.average(interPLV[2]),np.average(interPLV[3]),np.average(interPLV[4]),np.average(interPLV[0]),np.average(interPLV[1])
#
# _, p = scipy.stats.ttest_1samp(interAEC[2][0], 0.0)

# Checking structural - functional correlations within subjects
sfAEC = np.zeros(shape=(n, len(bands)))
sfPLV = np.zeros(shape=(n, len(bands)))
print("Calculating structural - functional correlations WITHIN subjects (PLV and AEC, independently)", end="")
for s in range(n):
   print(".", end="")
   sdir="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj"+str(s)+"\\"
   for b in range(len(bands)):
      AEC = np.loadtxt(sdir+bands[b]+"corramp.txt")
      PLV = np.loadtxt(sdir+bands[b]+"plv.txt")

      weights=np.loadtxt(sdir+"weights.txt")

      c1=np.zeros(shape=(2, 2145))
      c1[0,:]=AEC[np.triu_indices(66, 1)]
      c1[1,:]=weights[np.triu_indices(66, 1)]

      c2=np.zeros(shape=(2, 2145))
      c2[0,:]=PLV[np.triu_indices(66, 1)]
      c2[1,:]=weights[np.triu_indices(66, 1)]

      # Correlation between matrices
      sfAEC[s,b]=np.corrcoef(c1)[0,1]
      sfPLV[s,b]=np.corrcoef(c2)[0,1]

print(" %0.3f seconds.\n" % (time.time() - tic,))

del s, b, sdir, AEC, PLV, weights, c1, c2

# Check intrabands correlation¿?
bandsRaec = np.zeros(shape=(n, len(bands), len(bands)))
bandsRplv = np.zeros(shape=(n, len(bands), len(bands)))
print("Calculating correlations through bands WITHIN subjects (PLV and AEC, independently)", end="")
for s in range(n):
   print(".",end="")
   sdir="C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj"+str(s)+"\\"
   for b1 in range(len(bands)):
      for b2 in range(len(bands)):
         aec1=np.loadtxt(sdir+bands[b1]+"corramp.txt")
         aec2=np.loadtxt(sdir+bands[b2]+"corramp.txt")

         plv1 = np.loadtxt(sdir + bands[b1] + "plv.txt")
         plv2 = np.loadtxt(sdir + bands[b2] + "plv.txt")

         c1=np.zeros(shape=(2, 2145))
         c1[0,:]=aec1[np.triu_indices(66, 1)]
         c1[1,:]=aec2[np.triu_indices(66, 1)]

         c2=np.zeros(shape=(2, 2145))
         c2[0,:]=plv1[np.triu_indices(66, 1)]
         c2[1,:]=plv2[np.triu_indices(66, 1)]

         # Correlation between matrices
         bandsRaec[s,b1,b2]=np.corrcoef(c1)[0,1]
         bandsRplv[s,b1,b2]=np.corrcoef(c2)[0,1]

print(" %0.3f seconds.\n" % (time.time() - tic,))

del s, b1, b2, aec1, aec2, plv1, plv2, c1, c2, sdir
del n, tic, bands



# to plot just DMN

# indexes=np.array([1,7,22,24,25,34,40,55,57,58])
# DMNnames=names[indexes]
# DMN=matrix[indexes,:][:,indexes]
# fig = go.Figure(data=go.Heatmap(z=a, x=names, y=names, colorscale='Viridis'))
# fig.update_layout(title='Phase Locking Value')
# pio.write_html(fig, file="figures/PLV.html", auto_open=True)
#
#
# y = np.arange(35).reshape(5,7)
#
# y[1:5:2,::3]
#
# y[np.array([0,2,4]), np.array([0,1,2])]
#
# a=[[12,2,3],
#    [2,4,5],
#    [1,2,1]]
#
# b=np.triu(a,1)