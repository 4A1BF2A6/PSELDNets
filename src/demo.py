# import h5py

# with h5py.File('_hdf5/label/adpit/dev/official.h5', 'r') as f:
#     print("文件结构:")
#     def print_attrs(name, obj):
#         print(name)
#     f.visititems(print_attrs)



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Set font to ensure English display
plt.rcParams['font.family'] = ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 'sans-serif']

sr = 16000
duration = 2.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
low_freq = 300
high_freq = 3000
burst = np.zeros_like(t)
burst[::2000] = 1.0
signal = 0.6 * np.sin(2 * np.pi * low_freq * t) + 0.4 * np.sin(2 * np.pi * high_freq * t) + burst

f, times, Sxx = spectrogram(signal, fs=sr, nperseg=512, noverlap=256)
log_Sxx = 10 * np.log10(Sxx + 1e-10)
mel_range = f <= 8000
log_Sxx = log_Sxx[mel_range]
f = f[mel_range]

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(log_Sxx, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), f.min(), f.max()],
               cmap='viridis')

ax.add_patch(plt.Rectangle((0.5, 280), 0.8, 80, edgecolor='red', facecolor='none', lw=3))
ax.text(0.55, 750, 'Formant (Local Frequency Structure)', color='red', fontsize=12)

ax.add_patch(plt.Rectangle((0.12, 0), 0.1, 8000, edgecolor='cyan', facecolor='none', lw=3))
ax.text(0.13, 7700, 'Transient Changes (Temporal Continuity)', color='cyan', fontsize=12)

ax.add_patch(plt.Rectangle((1.2, 4000), 0.4, 2000, edgecolor='orange', facecolor='none', lw=3))
ax.text(1.22, 6000, 'Blurred Response Region\n(Symmetric Convolution Failure)', color='orange', fontsize=11)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
fig.colorbar(im, ax=ax, label="Log-Magnitude (dB)")
plt.title("Audio Spectrogram", fontsize=14)
plt.tight_layout()
plt.savefig("timefreq_heterogeneity_demo.png", dpi=120, bbox_inches='tight', facecolor='white')
