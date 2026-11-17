import pandas as pd

df=pd.read_csv("results/events.csv")

lines=["Earthquake Seismic Event Summary", "=====================\n"]

for _,row in df.iterrows():
    lines.append(
        f"Record: {row['record']}\n"
        f"  P-wave arrival (tP): {row['tP_s']:.2f} s\n"
        f"  S-wave arrival (tS): {row['tS_s']:.2f} s\n"
        f"  Epicentral distance (R): {row['R_km']:.2f} km\n"
        f"  Origin time (t0): {row['origin_t0_s']:.2f} s\n"
        f"  Peak Ground Velocity (PGV): {row['PGV_cm_s']:.4f} cm/s\n"
        f"  Estimated Magnitude (M): {row['M']:.2f}\n"
    )

closest=df.loc[df["R_km"].idxmin()]
strongest=df.loc[df["M"].idxmax()]

lines.append(
    f"Comparison:\n"
    f"  Closer earthquake: {closest['record']} ({closest['R_km']:.2f} km)\n"
    f"  Stronger earthquake: {strongest['record']} (M = {strongest['M']:.2f})\n"
)

report_text="\n".join(lines)
with open("results/report.txt","w") as f:
    f.write(report_text)

print(report_text)