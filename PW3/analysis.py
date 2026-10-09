
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==================== HEART DATA ====================

df = pd.read_csv("heart.csv")

print("Heart dataset shape:", df.shape)
print(df.head())

columns = ["age", "chol", "trestbps", "thalach"]
data = df[columns].dropna()

print("\nSummary statistics:")
print(data.describe())

# Distributions
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for col, ax in zip(columns, axes.flat):
    ax.hist(data[col], bins=20, edgecolor="black")
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("heart_distributions.png")
plt.close()

# Normality tests and Q-Q plots
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

print("\nNormality tests:")

for col, ax in zip(columns, axes.flat):
    values = data[col]
    sample = values.sample(min(len(values), 5000), random_state=1)
    statistic, p = stats.shapiro(sample)

    stats.probplot(values, dist="norm", plot=ax)
    ax.set_title(col)

    print(f"{col}: Shapiro p-value = {p:.5f}")

plt.tight_layout()
plt.savefig("heart_qqplots.png")
plt.close()

# ==================== HEART RATE COMPARISON ====================

healthy = df.loc[df["target"] == 0, "thalach"].dropna()
disease = df.loc[df["target"] == 1, "thalach"].dropna()

print("\nHeart rate comparison:")

for name, group in [("Healthy", healthy), ("Disease", disease)]:
    mean = group.mean()
    sem = stats.sem(group)
    margin = stats.t.ppf(0.975, len(group) - 1) * sem

    print(f"{name}: n={len(group)}, mean={mean:.2f}, "
          f"95% CI=({mean-margin:.2f}, {mean+margin:.2f})")

p1 = stats.shapiro(healthy).pvalue
p2 = stats.shapiro(disease).pvalue

if p1 >= 0.05 and p2 >= 0.05:
    result = stats.ttest_ind(healthy, disease, equal_var=False)
    print("Test: Welch's t-test")
else:
    result = stats.mannwhitneyu(healthy, disease, alternative="two-sided")
    print("Test: Mann-Whitney U test")

print("Test statistic:", result.statistic)
print("P-value:", result.pvalue)

means = [healthy.mean(), disease.mean()]
errors = []

for group in [healthy, disease]:
    margin = stats.t.ppf(0.975, len(group)-1) * stats.sem(group)
    errors.append(margin)

plt.figure(figsize=(6, 4))
plt.bar(["Healthy", "Disease"], means, yerr=errors,
        capsize=8, edgecolor="black")
plt.ylabel("Maximum heart rate (thalach)")
plt.title("Mean heart rate with 95% confidence intervals")
plt.tight_layout()
plt.savefig("heart_rate_comparison.png")
plt.close()

# ==================== AGE AND HEART RATE ====================

pair = df[["age", "thalach"]].dropna()

if (stats.shapiro(pair["age"].sample(min(len(pair), 5000),
                                     random_state=1)).pvalue >= 0.05
        and stats.shapiro(pair["thalach"].sample(min(len(pair), 5000),
                                                 random_state=1)).pvalue >= 0.05):
    corr = stats.pearsonr(pair["age"], pair["thalach"])
    method = "Pearson"
else:
    corr = stats.spearmanr(pair["age"], pair["thalach"])
    method = "Spearman"

print(f"\nAge-thalach {method} correlation: "
      f"{corr.statistic:.4f}, p-value={corr.pvalue:.6g}")

plt.figure(figsize=(6, 4))
plt.scatter(pair["age"], pair["thalach"], alpha=0.6)
plt.xlabel("Age")
plt.ylabel("Maximum heart rate")
plt.title("Age vs maximum heart rate")
plt.tight_layout()
plt.savefig("age_thalach.png")
plt.close()

# ==================== CHEMICAL DATA ====================

chem = pd.read_csv("chemicals_cancer.csv")
print("\nChemical dataset shape:", chem.shape)
print(chem.head())

print("\nNaive correlations with malignancy:")

for col in ["benzene", "cadmium"]:
    r, p = stats.pearsonr(chem[col], chem["malignancy"])
    print(f"{col}: r={r:.4f}, p-value={p:.6g}")

# Partial correlations controlling for pollution and age
def residuals(y, controls):
    X = np.column_stack([np.ones(len(controls)), controls])
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ coefficients

controls = chem[["pollution_index", "age"]].to_numpy()
malignancy = chem["malignancy"].to_numpy()

print("\nPartial correlations controlling for pollution and age:")

for col in ["benzene", "cadmium"]:
    chemical = chem[col].to_numpy()
    chemical_res = residuals(chemical, controls)
    malignancy_res = residuals(malignancy, controls)

    r, p = stats.pearsonr(chemical_res, malignancy_res)
    print(f"{col}: partial r={r:.4f}, p-value={p:.6g}")

# Compare correlations within a pollution range
subset = chem[(chem["pollution_index"] >= 40) &
              (chem["pollution_index"] <= 60)]

print("\nCorrelations for pollution index between 40 and 60:")
print("Number of patients:", len(subset))

for col in ["benzene", "cadmium"]:
    r, p = stats.pearsonr(subset[col], subset["malignancy"])
    print(f"{col}: r={r:.4f}, p-value={p:.6g}")

# Plot chemical associations
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for col, ax in zip(["benzene", "cadmium"], axes):
    ax.scatter(chem[col], chem["malignancy"], alpha=0.5)
    ax.set_xlabel(col)
    ax.set_ylabel("Malignancy")
    ax.set_title(f"{col} vs malignancy")

plt.tight_layout()
plt.savefig("chemical_correlations.png")
plt.close()

# ==================== BONUS: ENTROPY ====================

proportions = df["target"].value_counts(normalize=True)
entropy = -(proportions * np.log2(proportions)).sum()
maximum = np.log2(len(proportions))

print("\nTarget proportions:")
print(proportions)
print(f"Target entropy: {entropy:.4f} bits")
print(f"Maximum entropy: {maximum:.4f} bits")