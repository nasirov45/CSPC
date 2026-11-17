# CSPC - Computer Science for Physics and Chemistry

My coursework repository. Each practical is under PW<n>/Lab <X>/.

## Setup

Create the environment for a given lab:

```bash
conda env create -f PW<n>/Lab\ <X>/environment.yml
conda activate cspc
```

---

## PW1 - Lab A: Reproducible Foundations

**What I built:**

* Implemented pure-Python and NumPy versions of a radioactive decay simulation.
* Added tests for the initial atom count, negative decay rates, and the analytical decay law.

**Speed comparison (loop vs NumPy):**

* loop : 1.830334 s
* numpy : 0.000155 s
* speed-up: 11780.18 x faster

**Tests:** all passing? **Yes**

**Conclusion:**

* All three tests passed successfully.
* The NumPy implementation was much faster than the pure-Python loop for 200,000 atoms.
* I learned how vectorisation can significantly improve performance and how pytest can be used to test different parts of a simulation.

## PW1 --- Lab B

The observed decay data showed a decreasing number of counts over time, following the expected decay pattern.

The observed data matched the analytical decay law reasonably well, as the points followed a similar decreasing shape to the analytical curve.

The Snakemake pipeline takes the observed CSV data and `plot.py` as inputs and automatically generates `figure.png`, rebuilding it whenever an input file changes.


## PW2 — Lab A

### Motion from Tracking Data

The mean acceleration calculated from the free-fall data was **-8.58 m/s²**, which is close to the expected value of **-9.81 m/s²**.

The acceleration is much noisier than the position because differentiation amplifies small measurement errors and noise in the data.

After integrating the acceleration twice, the recovered position was close to the original position. The largest difference was approximately **0.785 m**, which is within the expected 1 metre difference.

### Bonus — 2D Trajectory

For the 2D trajectory data, the x and y coordinates were differentiated separately using `np.gradient` to calculate the velocity in each direction. The speed was then calculated from the x and y velocity components and plotted against time.


## PW2 — Lab B

### Optimisation in Chemistry

Three optimisation methods were compared: gradient descent, Newton's method, and SLSQP. For the simple convex function, all three methods converged to the same minimum at approximately **x = 3**.

For the more complicated function, the methods did not always give the same result. The starting point affected the result because the function has multiple stationary points. Newton's method can converge to a maximum as well as a minimum, so the second derivative must be checked. This shows that both the starting point and the optimisation algorithm matter for a more complicated landscape.

### Reaction Rate

The first-order reaction data was fitted using SLSQP. The fitted rate constant **k** was close to **0.25**, and the fitted exponential curve followed the measured concentration data.

### Chemical Equilibrium

For the reaction H₂ + I₂ ⇌ 2HI, the equilibrium extent was found using both Newton's method and SLSQP. Both methods gave approximately the same equilibrium value. The equilibrium composition was calculated from:

* H₂ = 1 − x mol
* I₂ = 1 − x mol
* HI = 2x mol

### Titration Equivalence Point

The pH slope was calculated using `np.gradient`. The equivalence point was found by locating the volume where the slope was largest. The equivalence point was approximately **50 mL**, where the pH curve changes most rapidly.


# PW3 — Data, Distributions, Testing, and Causation

## 1. Heart Disease Dataset

The Cleveland heart disease dataset contains 303 patients and 14 columns. The analysis uses age, cholesterol (`chol`), resting blood pressure (`trestbps`), maximum heart rate (`thalach`), and the disease indicator (`target`).

Source: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease).

### Distribution analysis

The four continuous variables were visualised using histograms and Q–Q plots. Summary statistics were calculated using pandas.

| Variable               |   Mean | Median | Standard deviation |
| ---------------------- | -----: | -----: | -----------------: |
| Age                    |  54.44 |     56 |               9.04 |
| Cholesterol            | 246.69 |    241 |              51.78 |
| Resting blood pressure | 131.69 |    130 |              17.60 |
| Maximum heart rate     | 149.61 |    153 |              22.88 |

### Normality testing

The Shapiro–Wilk test was used to assess normality. The null hypothesis is that the data follow a normal distribution. At a significance level of 0.05, all four variables reject this hypothesis.

| Variable               | Shapiro–Wilk p-value | Verdict                  |
| ---------------------- | -------------------: | ------------------------ |
| Age                    |              0.00607 | Not normally distributed |
| Cholesterol            |            < 0.00001 | Not normally distributed |
| Resting blood pressure |            < 0.00001 | Not normally distributed |
| Maximum heart rate     |              0.00007 | Not normally distributed |

The Q–Q plots and histograms were also used to inspect departures from normality. The statistical tests provide evidence against exact normality, although they do not indicate how severe the departures are.

## 2. Heart Rate and Heart Disease

The question was whether patients with heart disease have different maximum heart rates from healthy patients.

The two independent groups were compared using the Mann–Whitney U test because the normality analysis indicated departures from normality. The test compares the distributions of the two groups rather than directly testing their means.

| Group                        | Sample size | Mean maximum heart rate | 95% confidence interval |
| ---------------------------- | ----------: | ----------------------: | ----------------------: |
| Healthy (`target = 0`)       |         164 |                  158.38 |           155.42–161.34 |
| Heart disease (`target = 1`) |         139 |                  139.26 |           135.47–143.05 |

The test returned \(U = 16989.5\) and \(p = 1.86 \times 10^{-13}\).

Since the p-value is far below 0.05, the null hypothesis of identical group distributions is rejected. Patients with heart disease had a mean maximum heart rate approximately 19.12 beats per minute lower than healthy patients. The confidence intervals for the two means do not overlap, supporting the observed difference.

The confidence intervals quantify uncertainty in the estimated means; they are not the uncertainty measure used by the Mann–Whitney test itself.

## 3. Age and Maximum Heart Rate

Spearman's rank correlation was used because the variables did not satisfy the normality tests.

The correlation coefficient was \(\rho = -0.3916\), with \(p = 1.51 \times 10^{-12}\).

This indicates a moderate negative monotonic relationship: older patients tend to have lower maximum heart rates. The small p-value provides strong evidence that the observed association is not zero.

The scatter plot visualises this relationship.

## 4. Chemical Exposure and Malignancy

The synthetic dataset contains 1,000 patients and five variables: benzene, cadmium, pollution index, age, and malignancy.

### Naive analysis

Pearson correlation coefficients were calculated between each chemical and malignancy.

| Chemical | Correlation |                  p-value |
| -------- | ----------: | -----------------------: |
| Benzene  |      0.3847 | \(1.28 \times 10^{-36}\) |
| Cadmium  |      0.8800 |          Approximately 0 |

The naive analysis suggests that cadmium has a much stronger positive association with malignancy than benzene. If correlation were interpreted as causation, cadmium would appear to be the main cause.

### Accounting for confounding

A confounding variable can influence both a chemical's measured concentration and malignancy, creating a misleading association.

To investigate this, partial correlations were calculated while controlling for pollution index and age. This removes the linear relationships with those control variables before measuring the association between each chemical and malignancy.

| Chemical | Partial correlation |                   p-value |
| -------- | ------------------: | ------------------------: |
| Benzene  |              0.7595 | \(1.47 \times 10^{-188}\) |
| Cadmium  |              0.0069 |                    0.8278 |

The results change substantially after adjustment. Benzene has a strong positive association with malignancy, whereas cadmium's adjusted association is practically zero and statistically insignificant.

As an additional check, the analysis was restricted to patients with pollution index values between 40 and 60. Within this subgroup of 201 patients, benzene had a correlation of 0.6979 with malignancy, while cadmium had a correlation of 0.3132. Both remained positive in this restricted sample, illustrating that stratification alone does not completely remove confounding.

### Conclusion: correlation versus causation

The naive analysis made cadmium look like the more important chemical because its raw correlation with malignancy was 0.8800. After controlling for pollution index and age, this association disappeared, while benzene's association remained strong.

Therefore, benzene is the likely causal chemical in this synthetic dataset, while cadmium's strong raw association is explained by confounding. This conclusion is supported by the adjusted statistical results and the dataset's intended design, although statistical adjustment alone cannot establish causation in real observational data.

## 5. Bonus: Target Entropy

The target proportions were:

* Healthy patients: 54.13%
* Patients with heart disease: 45.87%

Shannon entropy was calculated using

$$
H(X)=-\sum_i p_i\log_2(p_i).
$$

The entropy was 0.9951 bits, close to the maximum of 1 bit for two equally likely categories.

This means the target is nearly balanced, so guessing the majority category alone would achieve only about 54.13% accuracy.

## 6. Files and Reproducibility

The `analysis.py` script loads both datasets, calculates summary statistics, performs normality tests and correlation analyses, compares the heart-rate groups, and generates the plots.

Generated plots:

* `heart_distributions.png`
* `heart_qqplots.png`
* `heart_rate_comparison.png`
* `age_thalach.png`
* `chemical_correlations.png`

The analysis can be reproduced by activating the `cspc` Conda environment and running `python analysis.py` from the `PW3` directory.


# PW4 — Machine Learning: Supervised and Unsupervised

## Session A — Concrete Strength

### 1. Dataset and Data Preparation

The Concrete Compressive Strength dataset was downloaded from the UCI Machine Learning Repository. It contains 1,030 observations, eight input features, and one target variable: concrete compressive strength in MPa.

The input features describe the concrete mixture and curing age: cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age.

### 2. Data Cleaning and Exploration

The dataset initially contained 1,030 rows. A total of 25 duplicate rows were found and removed, leaving 1,005 unique observations. No missing values were found in any column.

Removing duplicates before splitting the data helps prevent identical observations from appearing in both the training and test sets, which could make the model's evaluation misleading.

**Scatter plot observations**

The *Strength vs Age* plot shows that concrete strength generally increases with curing age. The increase is especially rapid during the first 28 days. After that, strength growth becomes slower and tends to stabilize. The points are concentrated around standard curing ages, with noticeable variation in strength at each age.

The *Strength vs Cement* plot shows a clear positive relationship between cement content and compressive strength. In general, higher cement content corresponds to higher concrete strength. However, the points are spread vertically at similar cement levels, indicating that other factors, such as water content, aggregates, additives, and curing age, also influence strength.

### 3. Train/Test Split

The cleaned dataset was divided into training and testing sets using an 80/20 split. The training set contains 804 observations, while the test set contains 201 observations.

The `random_state=42` parameter ensures that the split is reproducible. The training data is used to fit the models, while the test data is kept separate for evaluating their performance on unseen observations.

**Residual analysis**

The residual plot shows that residuals are mostly positive at some curing ages and negative at others, rather than being randomly distributed around zero. Positive residuals indicate underprediction, while negative residuals indicate overprediction. This systematic pattern suggests that the linear model does not fully capture the relationship between curing age and concrete strength, which may be nonlinear.

### 5. Polynomial Regression

Polynomial regression was used to investigate whether nonlinear relationships between concrete properties and compressive strength could improve prediction accuracy.

Four models with polynomial degrees 1, 2, 3 and 4 were trained and evaluated. `PolynomialFeatures` generated additional features, `StandardScaler` normalized them, and `LinearRegression` fitted the models.

**Results:**

| Degree | Features | Train RMSE | Test RMSE | Train R² | Adjusted Train R² | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 10.0018 | 11.1913 | 0.6099 | 0.6059 | 0.5802 |
| 2 | 44 | 6.9892 | 8.3080 | 0.8095 | 0.7984 | 0.7686 |
| 3 | 164 | 4.3961 | 5.9515 | 0.9246 | 0.9053 | 0.8813 |
| 4 | 494 | 2.3339 | 39.4096 | 0.9788 | 0.9448 | -4.2061 |

**Model selection**

Degree 3 was selected as the best model because it achieved the lowest test RMSE (5.9515) and the highest test R² (0.8813). It explains approximately 88% of the variation in concrete compressive strength on the test set.

Degree 4 showed severe overfitting. Although its training RMSE was only 2.3339, its test RMSE increased to 39.4096 and its test R² became negative. This happened because the model generated 494 features from only 804 training observations, making it excessively complex.

Adjusted R² penalizes unnecessary features, but it does not completely prevent overfitting. Test-set performance is therefore essential when selecting a model.

### A6 – Classification: Predicting Concrete Strength

The target variable was converted into a binary classification problem: class 1 represents concrete with strength of at least 40 MPa, while class 0 represents concrete with strength below 40 MPa.

The dataset was split into training (80%) and testing (20%) sets using `random_state=42` and stratification. Both sets maintained the same class proportions: 64.18% class 0 and 35.82% class 1.

A KNN classifier with StandardScaler was tested using K values of 1, 3, 5, 7, 15, 25, 51 and 101. The best test accuracy was achieved with **K=7 (83.58%)**. K=1 showed very high training accuracy (99.75%) but lower test accuracy, suggesting overfitting.

Scaling improved the test accuracy from 82.09% to 83.58%, demonstrating the importance of feature scaling for distance-based algorithms.

The model was also evaluated using five different train/test splits. The mean test accuracy was 82.29%, with a standard deviation of 1.85 percentage points and an accuracy range of 4.98 percentage points. This indicates relatively stable performance across different splits.

### Decision Tree Classification

A Decision Tree classifier using entropy was tested with maximum depths of 1, 2, 3, 4, 5, 6, 8 and unlimited depth.

Test accuracy generally increased with tree depth. The unlimited tree achieved the highest test accuracy of 88.06%, but its training accuracy was 99.75%, indicating possible overfitting. It also contained 80 leaves, making it more complex and less interpretable.

A depth of 5 was selected as a compromise between accuracy and interpretability. It achieved 84.58% test accuracy with 26 leaves.

The root feature of the selected tree was **age**, with the first question being whether curing age is less than or equal to 42 days. Samples with age up to 42 days are further divided based on cement content, while older samples are divided based on superplasticizer content.

This structure is physically reasonable because curing age and mixture composition influence concrete strength.

### A7 – Classification Model Evaluation

The selected KNN (K=7) and Decision Tree (maximum depth=5) models were evaluated using confusion matrices, accuracy, precision, recall and F1-score.

The KNN model achieved 83.58% accuracy, 78.26% precision, 75.00% recall and 76.60% F1-score.

The Decision Tree achieved 84.58% accuracy, 77.33% precision, 80.56% recall and 78.91% F1-score.

Overall, the Decision Tree performed slightly better, achieving higher accuracy, recall and F1-score, while KNN had slightly higher precision.

In a construction-site application, a dangerous error is a false negative: predicting that concrete reaches the required 40 MPa strength when its actual strength is below 40 MPa. This could lead to inadequate concrete being accepted for structural use. Therefore, reducing false negatives is important.

### Threshold Analysis

The KNN model was evaluated using probability thresholds of 0.3, 0.5, 0.7 and 0.9.

As the threshold increased, the number of false positives decreased from 23 to 1, while the number of false negatives increased from 11 to 54.

A threshold of 0.9 was selected for the safety-focused example because it achieved 94.74% precision and only one false positive. This reduces the risk of classifying concrete with actual strength below 40 MPa as acceptable.

However, recall decreased to 25%, meaning that many samples that actually meet the strength requirement would be rejected. This demonstrates the trade-off between safety and efficiency.

### A8 – Feature Importance

Feature importance was compared using linear regression coefficients, standardized linear regression coefficients, Decision Tree feature importance, and permutation importance for KNN and Decision Tree.

The raw linear regression coefficients cannot be directly compared by magnitude because the features have different units and scales. After standardization, cement had the largest coefficient magnitude (12.1188), followed by slag (8.4373) and age (6.8945). Water had a negative standardized coefficient (-2.8279), indicating a negative relationship in the linear model.

For the Decision Tree, the most important features were age (0.3126), cement (0.2525), superplasticizer (0.1645), water (0.0952), and fine aggregate (0.0905). Age was therefore the most important feature for the tree model.

Permutation importance using F1 score showed that the KNN model relied most on age (0.1757), cement (0.1646), and fine aggregate (0.0808). For the Decision Tree, the most important features were superplasticizer (0.1881), cement (0.1698), and age (0.1648).

Overall, age and cement were consistently among the most important features across the different methods, while the importance of other features varied depending on the model and importance measure. This shows that different models can rely on the input features in different ways.

### B1 – Wine dataset exploration

The Wine dataset contains 178 samples and 13 numerical chemical measurements. It originates from chemical analyses of wines grown in the same Italian region by three different cultivators. The dataset has three cultivar classes: class 0 (59 samples), class 1 (71 samples), and class 2 (48 samples). There are no missing values.

The measurement with the largest standard deviation is proline (314.91), followed by magnesium (14.30). Nonflavanoid phenols has the smallest standard deviation (0.12). Proline also has the largest range (278–1680), while nonflavanoid phenols has the smallest range (0.13–0.66).

The features have very different numerical scales. This is important because measurements with larger values and variances may dominate PCA when the original data is not standardized.

### B2 – PCA on raw data

PCA was applied to the original, unscaled measurements. The first principal component (PC1) explains approximately 99.81% of the total variance.

The PC1 loading for proline is 0.999823, while the loadings of all other measurements are close to zero. This indicates that PC1 is almost entirely dominated by proline.

We should not interpret this as evidence that proline represents the main chemical difference between the wines. Proline has a much larger numerical scale and variance than the other features. Therefore, raw PCA is strongly influenced by differences in measurement scales. Standardization is necessary before performing a more balanced PCA analysis.

### B3 – PCA on standardized data

The features were standardized using StandardScaler. The resulting means are approximately zero and standard deviations are equal to one.

After standardization, PC1 explains 36.20% of the variance and PC2 explains 19.21%. Together, they explain 55.41%. Eight principal components are required to retain at least 90% of the total variance.

PC1 has strong positive loadings for flavanoids, total phenols, and OD280/OD315, while malic acid, nonflavanoid phenols, and alcalinity of ash have negative loadings. PC2 is mainly associated with positive loadings for color intensity and alcohol, while hue has a negative loading.

The PC1–PC2 scatter plot shows approximately three broad groups, although some overlap exists. This suggests that the wines may have distinguishable chemical profiles, but the first two components do not preserve all the information in the original dataset.

### B4 – K-Means Clustering

To evaluate the sensitivity of K-Means to random initialization, the algorithm was fitted with $k = 3$ and `n_init=1` across eight different random seeds (`random_state` from 0 to 7).

| Random state | Inertia |
| ---: | ---: |
| 0 | 1277.93 |
| 1 | 1278.76 |
| 2 | 1277.93 |
| 3 | 1277.93 |
| 4 | 1279.97 |
| 5 | 1278.76 |
| 6 | 1282.46 |
| 7 | 1278.76 |

The inertia values vary slightly across seeds because K-Means converges to local minima depending on the initial centroid positions. Setting `n_init=10` runs the algorithm ten times from different random starting positions and keeps the solution with the lowest inertia (**1277.93**), reducing the risk of a suboptimal clustering.

The elbow method was then applied by fitting K-Means (`n_init=10`, `random_state=0`) for $k = 1$ to $10$:

| Number of clusters ($k$) | Inertia |
| ---: | ---: |
| 1 | 2314.00 |
| 2 | 1659.01 |
| 3 | 1277.93 |
| 4 | 1180.74 |
| 5 | 1110.37 |
| 6 | 1044.47 |
| 7 | 995.99 |
| 8 | 944.56 |
| 9 | 913.20 |
| 10 | 864.58 |

**Choice of $k$**

$k = 3$ was selected from the elbow curve. Inertia drops sharply from $k = 1$ to $k = 2$ (by 654.99) and from $k = 2$ to $k = 3$ (by 381.08), after which the decrease becomes much more gradual (dropping by only 97.19 from $k = 3$ to $k = 4$).

$k$ cannot be chosen simply by minimizing inertia because inertia decreases monotonically as $k$ increases, reaching zero when $k$ equals the number of samples (178), where each wine forms its own cluster.

### B5 – Cluster Profiles

With $k = 3$, the clusters contain **65** wines (Cluster 0), **51** wines (Cluster 1), and **62** wines (Cluster 2). The cluster centres were transformed back to real units using `scaler.inverse_transform`:

| Measurement | Cluster 0 (65) | Cluster 1 (51) | Cluster 2 (62) |
| :--- | ---: | ---: | ---: |
| Alcohol | 12.25 | 13.13 | 13.68 |
| Malic acid | 1.90 | 3.31 | 2.00 |
| Ash | 2.23 | 2.42 | 2.47 |
| Alcalinity of ash | 20.06 | 21.24 | 17.46 |
| Magnesium | 92.74 | 98.67 | 107.97 |
| Total phenols | 2.25 | 1.68 | 2.85 |
| Flavanoids | 2.05 | 0.82 | 3.00 |
| Nonflavanoid phenols | 0.36 | 0.45 | 0.29 |
| Proanthocyanins | 1.62 | 1.15 | 1.92 |
| Color intensity | 2.97 | 7.23 | 5.45 |
| Hue | 1.06 | 0.69 | 1.07 |
| OD280/OD315 | 2.80 | 1.70 | 3.16 |
| Proline | 510.17 | 619.06 | 1100.23 |

**Cluster descriptions**

* **Cluster 0:** Light-bodied, pale wines with the lowest alcohol (12.25%), lowest color intensity (2.97), lowest proline (510.17), and moderate phenolic content.
* **Cluster 1:** High-acid, deeply colored wines characterized by the highest malic acid (3.31), highest color intensity (7.23), and lowest hue (0.69), but the lowest concentrations of flavanoids (0.82), total phenols (1.68), and OD280/OD315 (1.70).
* **Cluster 2:** Full-bodied, phenolic-rich wines with the highest alcohol (13.68%), nearly double the proline (1100.23), and the highest levels of flavanoids (3.00), total phenols (2.85), and OD280/OD315 (3.16).

### B6 – Comparison with True Cultivars

The discovered clusters were compared against the true cultivar labels using crosstabs and agreement scores for three feature representations ($k = 3$):

| Representation | Agreement | Correctly grouped |
| :--- | ---: | ---: |
| Scaled (13 features) | 0.9663 | 172 / 178 |
| Raw unscaled (13 features) | 0.7022 | 125 / 178 |
| Scaled PCA (PC1 & PC2) | 0.9663 | 172 / 178 |

**Analysis**

Raw unscaled K-Means achieved a much lower agreement (0.7022) because Euclidean distances are dominated by high-variance features such as proline and magnesium, ignoring important chemical differences in phenols, flavanoids, and color intensity.

Standardizing all 13 features puts every measurement on equal footing, increasing agreement to 0.9663 (separating 100% of cultivar 0 and cultivar 2 cleanly, with only 6 wines from cultivar 1 overlapping). K-Means on the first two standardized principal components achieved the exact same agreement (0.9663), showing that although PC1 and PC2 retain 55.41% of total variance, they already capture the main chemical separation among the three cultivars.

Because K-Means never used the true labels during fitting, this high agreement indicates that the three grape cultivars possess distinct, naturally separable chemical profiles.