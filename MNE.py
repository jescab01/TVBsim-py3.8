import os.path
import mne
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import envelope_correlation

import numpy as np
import matplotlib.pyplot as plt

# Compute envelope correlations in source space
#
# Compute envelope correlations of orthogonalized activity 1 2 in source space using resting state CTF data.

data_path = mne.datasets.brainstorm.bst_resting.data_path()

subjects_dir = os.path.join(data_path, 'subjects')
subject = 'bst_resting'
trans = os.path.join(data_path, 'MEG', 'bst_resting', 'bst_resting-trans.fif')
src = os.path.join(subjects_dir, subject, 'bem', subject + '-oct-6-src.fif')
bem = os.path.join(subjects_dir, subject, 'bem', subject + '-5120-bem-sol.fif')
raw_fname = os.path.join(data_path, 'MEG', 'bst_resting','subj002_spontaneous_20111102_01_AUX.ds')

# Here we do some things in the name of speed, such as crop (which will hurt SNR) and downsample.
# Then we compute SSP projectors and apply them.
raw = mne.io.read_raw_ctf(raw_fname, verbose='error')
raw.crop(0, 60).load_data().pick_types(meg=True, eeg=False).resample(80)
raw.apply_gradient_compensation(3)
projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2)
projs_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='MLT31-4407')
raw.info['projs'] += projs_ecg
raw.info['projs'] += projs_eog
raw.apply_proj()
cov = mne.compute_raw_covariance(raw)  # compute before band-pass of interest


# Now we band-pass filter our data and create epochs.
raw.filter(14, 30)
events = mne.make_fixed_length_events(raw, duration=5.)
epochs = mne.Epochs(raw, events=events, tmin=0, tmax=5., baseline=None, reject=dict(mag=8e-13), preload=True)
del raw

# Compute forward and inverse
src = mne.read_source_spaces(src)
fwd = mne.make_forward_solution(epochs.info, trans, src, bem)
inv = make_inverse_operator(epochs.info, fwd, cov)
del fwd, src

# Compute label time series and do envelope correlation
labels = mne.read_labels_from_annot(subject, 'aparc_sub', subjects_dir=subjects_dir)
epochs.apply_hilbert()  # faster to apply in sensor space
stcs = apply_inverse_epochs(epochs, inv, lambda2=1. / 9., pick_ori='normal', return_generator=True)
label_ts = mne.extract_label_time_course(stcs, labels, inv['src'], return_generator=True)
corr = envelope_correlation(label_ts, verbose=True)

# let's plot this matrix
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(corr, cmap='viridis', clim=np.percentile(corr, [5, 95]))
fig.tight_layout()


# Compute the degree and plot it
threshold_prop = 0.15  # percentage of strongest edges to keep in the graph
degree = mne.connectivity.degree(corr, threshold_prop=threshold_prop)
stc = mne.labels_to_stc(labels, degree)
stc = stc.in_label(mne.Label(inv['src'][0]['vertno'], hemi='lh') + mne.Label(inv['src'][1]['vertno'], hemi='rh'))
brain = stc.plot(
    clim=dict(kind='percent', lims=[75, 85, 95]), colormap='gnuplot',
    subjects_dir=subjects_dir, views='dorsal', hemi='both',
    smoothing_steps=25, time_label='Beta band')