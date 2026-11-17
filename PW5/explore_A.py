import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, spectrogram

os.makedirs("results",exist_ok=True)

# 1. Load and inspect the record
d=pd.read_csv("data/seismogram_A.csv")
t=d["time_s"].to_numpy()
Z=d["Z"].to_numpy()
N=d["N"].to_numpy()
E=d["E"].to_numpy()

fs=1.0/(t[1]-t[0])
print(f"Sampling rate: {fs} Hz, Duration: {t[-1]:.2f} s")

# Plot 3 channels in mm/s
fig,axes=plt.subplots(3,1,sharex=True,figsize=(10,7))
channels=[(Z,"Z (Vertical)"),(N,"N (North-South)"),(E,"E (East-West)")]
for ax,(sig,label) in zip(axes,channels):
    ax.plot(t,sig*1000.0,linewidth=0.6)
    ax.set_ylabel("Velocity (mm/s)")
    ax.set_title(label)
    ax.grid(True,linestyle="--",alpha=0.4)
axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig("results/1_raw_channels.png",dpi=150)
plt.close(fig)

# Zoom on first 60 s
quiet=t<60.0
fig,axes=plt.subplots(3,1,sharex=True,figsize=(10,6))
for ax,(sig,label) in zip(axes,channels):
    ax.plot(t[quiet],sig[quiet]*1000.0,linewidth=0.6)
    ax.set_ylabel("Velocity (mm/s)")
    ax.set_title(f"{label} — First 60 s")
    ax.grid(True,linestyle="--",alpha=0.4)
axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig("results/1_quiet_zoom.png",dpi=150)
plt.close(fig)

# 2. FFT Spectrum
n_samples=len(Z)
X_fft=np.fft.fft(Z)
freqs=np.fft.fftfreq(n_samples,d=1.0/fs)
amp=np.abs(X_fft)/n_samples*2.0

fig,axes=plt.subplots(2,1,figsize=(10,6))
axes[0].plot(np.arange(n_samples),amp,linewidth=0.6)
axes[0].set_xlabel("Array index")
axes[0].set_ylabel("Amplitude (m/s)")
axes[0].set_title("FFT Amplitude vs Array Index")

axes[1].plot(freqs,amp,linewidth=0.6)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude (m/s)")
axes[1].set_title("FFT Amplitude vs Frequency (fftfreq)")
fig.tight_layout()
fig.savefig("results/2a_fft_index_vs_freq.png",dpi=150)
plt.close(fig)

# 2(b) Log-log spectrum: quiet (first 60 s) vs whole record
Z_quiet=Z[quiet]
n_q=len(Z_quiet)
f_q=np.fft.fftfreq(n_q,d=1.0/fs)
amp_q=np.abs(np.fft.fft(Z_quiet))/n_q*2.0

pos=freqs>0
pos_q=f_q>0

fig,ax=plt.subplots(figsize=(9,5))
ax.loglog(freqs[pos],amp[pos],label="Whole record (300 s)",alpha=0.8)
ax.loglog(f_q[pos_q],amp_q[pos_q],label="Quiet part (first 60 s)",alpha=0.85)
ax.axvline(0.5,color="red",linestyle="--",label="Band-pass cut-offs (0.5 & 10 Hz)")
ax.axvline(10.0,color="red",linestyle="--")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude (m/s)")
ax.set_title("Log-Log Amplitude Spectrum: Quiet Background vs Earthquake")
ax.legend()
ax.grid(True,which="both",linestyle="--",alpha=0.4)
fig.tight_layout()
fig.savefig("results/2b_loglog_spectrum.png",dpi=150)
plt.close(fig)

# 3. Inverse FFT & Band-pass filter
print("IFFT(FFT(Z)).real equals Z:",np.allclose(np.fft.ifft(np.fft.fft(Z)).real,Z))

def bandpass(x,fs,a,b):
    n=len(x)
    X=np.fft.fft(x)
    f=np.fft.fftfreq(n,1.0/fs)
    X[(np.abs(f)<a)|(np.abs(f)>b)]=0
    return np.fft.ifft(X).real

Zf=bandpass(Z,fs,0.5,10.0)
Nf=bandpass(N,fs,0.5,10.0)
Ef=bandpass(E,fs,0.5,10.0)

# 3(c) What was removed: Z - Zf
fig,ax=plt.subplots(figsize=(10,4))
ax.plot(t,(Z-Zf)*1000.0,linewidth=0.6)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity (mm/s)")
ax.set_title("Removed Signal (Z - Z_filtered)")
ax.grid(True,linestyle="--",alpha=0.4)
fig.tight_layout()
fig.savefig("results/3c_removed_signal.png",dpi=150)
plt.close(fig)

# 4. Envelopes
def get_envelope(x,fs,win_s):
    w=int(round(win_s*fs))
    return np.convolve(np.abs(x),np.ones(w)/w,mode="same")

env_Z_05=get_envelope(Zf,fs,0.5)
env_Z_005=get_envelope(Zf,fs,0.05)
env_Z_5=get_envelope(Zf,fs,5.0)

fig,ax=plt.subplots(figsize=(10,4))
ax.plot(t,env_Z_005*1000.0,label="Window = 0.05 s",alpha=0.7)
ax.plot(t,env_Z_05*1000.0,label="Window = 0.5 s",linewidth=1.5)
ax.plot(t,env_Z_5*1000.0,label="Window = 5.0 s",linewidth=1.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Envelope (mm/s)")
ax.set_title("Effect of Smoothing Window Length on Z Envelope")
ax.legend()
ax.grid(True,linestyle="--",alpha=0.4)
fig.tight_layout()
fig.savefig("results/4_envelopes_comparison.png",dpi=150)
plt.close(fig)

# 5. Arrival times tP and tS
Hf=np.sqrt(Nf**2+Ef**2)
env_H=get_envelope(Hf,fs,0.5)

tP_idx=np.argmax(env_Z_05>0.05*np.max(env_Z_05))
tP=t[tP_idx]

cond_S=(t>(tP+2.0))&(env_H>0.10*np.max(env_H))
tS_idx=np.argmax(cond_S)
tS=t[tS_idx]

peaks,_=find_peaks(env_H)
t_max_H=t[peaks[np.argmax(env_H[peaks])]]

print(f"tP = {tP:.2f} s, tS = {tS:.2f} s, Horizontal max time = {t_max_H:.2f} s")

fig,axes=plt.subplots(3,1,sharex=True,figsize=(10,7))
filt_channels=[(Zf,"Z Filtered (0.5-10 Hz)"),(Nf,"N Filtered (0.5-10 Hz)"),(Ef,"E Filtered (0.5-10 Hz)")]
for ax,(sig,label) in zip(axes,filt_channels):
    ax.plot(t,sig*1000.0,linewidth=0.6)
    ax.axvline(tP,color="green",linestyle="--",label=f"tP = {tP:.2f} s")
    ax.axvline(tS,color="red",linestyle="--",label=f"tS = {tS:.2f} s")
    ax.set_ylabel("Velocity (mm/s)")
    ax.set_title(label)
    ax.legend(loc="upper left")
    ax.grid(True,linestyle="--",alpha=0.4)
axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig("results/5_arrivals.png",dpi=150)
plt.close(fig)

# 6. Distance & origin time
vP,vS=6.2,3.6
R=(tS-tP)/(1.0/vS-1.0/vP)
R_max=(t_max_H-tP)/(1.0/vS-1.0/vP)
dR_1s=1.0/(1.0/vS-1.0/vP)
t0=tP-R/vP
print(f"R = {R:.2f} km (using max: {R_max:.2f} km), +1s uncertainty -> +{dR_1s:.2f} km, origin t0 = {t0:.2f} s")

# 7. Spectrogram of raw Z up to 10 Hz
f_spec,t_spec,Sxx=spectrogram(Z,fs=fs,nperseg=256)
mask_10=f_spec<=10.0

fig,ax=plt.subplots(figsize=(10,5))
pcm=ax.pcolormesh(t_spec,f_spec[mask_10],10.0*np.log10(Sxx[mask_10]+1e-20),shading="auto")
ax.axvline(tP,color="white",linestyle="--",label="P arrival")
ax.axvline(tS,color="cyan",linestyle="--",label="S arrival")
fig.colorbar(pcm,ax=ax,label="Power Spectral Density (dB)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Spectrogram of Raw Z Channel (0-10 Hz)")
ax.legend()
fig.tight_layout()
fig.savefig("results/7_spectrogram.png",dpi=150)
plt.close(fig)

# 8. Magnitude
PGV_cms=np.max(np.sqrt(N**2+E**2))*100.0
M=(np.log10(PGV_cms)+np.log10(R)+1.0)/0.55
print(f"PGV = {PGV_cms:.4f} cm/s, Magnitude M = {M:.2f}")