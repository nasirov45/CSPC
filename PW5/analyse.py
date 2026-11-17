import os
import numpy as np
import pandas as pd

def bandpass(x,fs,a,b):
    n=len(x)
    X=np.fft.fft(x)
    f=np.fft.fftfreq(n,1.0/fs)
    X[(np.abs(f)<a)|(np.abs(f)>b)]=0
    return np.fft.ifft(X).real

def envelope(x,fs,win_s):
    w=int(round(win_s*fs))
    return np.convolve(np.abs(x),np.ones(w)/w,mode="same")

def analyse(csv_file):
    d=pd.read_csv(csv_file)
    t=d["time_s"].to_numpy()
    Z=d["Z"].to_numpy()
    N=d["N"].to_numpy()
    E=d["E"].to_numpy()

    fs=1.0/(t[1]-t[0])
    vP,vS=6.2,3.6

    # Band-pass filter (0.5 - 10 Hz)
    Zf=bandpass(Z,fs,0.5,10.0)
    Nf=bandpass(N,fs,0.5,10.0)
    Ef=bandpass(E,fs,0.5,10.0)

    # Envelopes (0.5 s window)
    Hf=np.sqrt(Nf**2+Ef**2)
    env_Z=envelope(Zf,fs,0.5)
    env_H=envelope(Hf,fs,0.5)

    # Arrival times
    tP=t[np.argmax(env_Z>0.05*np.max(env_Z))]
    cond_S=(t>(tP+2.0))&(env_H>0.10*np.max(env_H))
    tS=t[np.argmax(cond_S)]

    # Distance and origin time
    R=(tS-tP)/(1.0/vS-1.0/vP)
    t0=tP-R/vP

    # Magnitude from raw horizontal PGV in cm/s
    PGV_cms=np.max(np.sqrt(N**2+E**2))*100.0
    M=(np.log10(PGV_cms)+np.log10(R)+1.0)/0.55

    return {
        "record":os.path.basename(csv_file),
        "tP_s":round(float(tP),2),
        "tS_s":round(float(tS),2),
        "R_km":round(float(R),2),
        "origin_t0_s":round(float(t0),2),
        "PGV_cm_s":round(float(PGV_cms),4),
        "M":round(float(M),2)
    }

if __name__=="__main__":
    os.makedirs("results",exist_ok=True)
    records=["data/seismogram_A.csv","data/seismogram_B.csv"]
    rows=[analyse(r) for r in records]
    df_out=pd.DataFrame(rows)
    df_out.to_csv("results/events.csv",index=False)
    print(df_out.to_string(index=False))