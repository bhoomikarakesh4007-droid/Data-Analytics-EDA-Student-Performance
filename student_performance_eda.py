"""
Exploratory Data Analysis (EDA) and Data Visualization on Student Academic Performance
Dataset: Student Performance Factors (Kaggle / GitHub raw CSV mirror)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure clean visual style
sns.set_theme(style="whitegrid")
os.makedirs("outputs", exist_ok=True)

# ==============================================================================
# Step 1 & 2: Load the Real Dataset
# ==============================================================================
DATASET_URL = (
    "https://raw.githubusercontent.com/fiaaha/"
    "StudentPerformanceFactors/main/StudentPerformanceFactors.csv"
)
df = pd.read_csv(DATASET_URL)
print("Dataset loaded successfully!\n")

# ==============================================================================
# Step 3: Inspect Dataset Shape, Columns, Data Types, and Sample Rows
# ==============================================================================
print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df[["Hours_Studied", "Attendance", "Previous_Scores", "Exam_Score"]].head())

# ==============================================================================
# Step 4: Basic Data Cleaning and Handling Missing Values
# ==============================================================================
print("\nMissing Values Before Cleaning:")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0])

print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# Impute missing categorical values with column mode
df_clean = df.copy()
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

print(f"Total Missing Values After Cleaning: {df_clean.isnull().sum().sum()}")

# Check for out-of-range values in Exam_Score (> 100)
anomalies = df_clean[df_clean["Exam_Score"] > 100]
print(f"Records with Exam_Score > 100: {len(anomalies)}")
if len(anomalies) > 0:
    print(anomalies[["Hours_Studied", "Attendance", "Previous_Scores", "Exam_Score"]])

# ==============================================================================
# Step 5: Select Relevant Numerical Columns for Analysis
# ==============================================================================
# Mapping to problem statement variables:
#   Hours_Studied   -> Study Hours (hours/week)
#   Attendance      -> Attendance Percentage (%)
#   Previous_Scores -> Internal / Previous Marks (out of 100)
#   Exam_Score      -> Final Examination Score (out of 100)
selected_cols = ["Hours_Studied", "Attendance", "Previous_Scores", "Exam_Score"]
eda_df = df_clean[selected_cols]

# ==============================================================================
# Step 6: Calculate Mean, Median, Mode, Variance, and Standard Deviation
# ==============================================================================
stats_summary = pd.DataFrame({
    "Mean": eda_df.mean(),
    "Median": eda_df.median(),
    "Mode": eda_df.mode().iloc[0],
    "Variance": eda_df.var(),
    "Std_Dev": eda_df.std(),
    "Min": eda_df.min(),
    "Max": eda_df.max(),
})
print("\nDescriptive Statistics Summary:")
print(stats_summary.round(2))

# IQR Outlier Detection Summary
print("\nIQR Outlier Analysis:")
for col in selected_cols:
    q1 = eda_df[col].quantile(0.25)
    q3 = eda_df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = eda_df[(eda_df[col] < lower) | (eda_df[col] > upper)][col]
    print(
        f"  {col:16s} | Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}, "
        f"Fence=[{lower:.1f}, {upper:.1f}], Outliers={len(outliers)}"
    )

# ==============================================================================
# Step 7: Histogram of Final Examination Scores
# ==============================================================================
plt.figure(figsize=(7.5, 4.5))
sns.histplot(eda_df["Exam_Score"], bins=22, kde=True, color="#2B6CB0", edgecolor="white")
mean_score = eda_df["Exam_Score"].mean()
median_score = eda_df["Exam_Score"].median()
plt.axvline(mean_score, color="#E53E3E", linestyle="--", linewidth=1.8, label=f"Mean = {mean_score:.2f}")
plt.axvline(median_score, color="#38A169", linestyle="-.", linewidth=1.8, label=f"Median = {median_score:.2f}")
plt.title("Distribution of Final Examination Scores (Exam_Score)", fontsize=12, fontweight="bold")
plt.xlabel("Final Examination Score", fontsize=10)
plt.ylabel("Number of Students", fontsize=10)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/fig1_exam_score_histogram.png", dpi=300)
plt.show()

# ==============================================================================
# Step 8: Boxplot to Identify Possible Outliers in Student Scores
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2))
sns.boxplot(y=eda_df["Exam_Score"], ax=axes[0], color="#63B3ED", width=0.45)
axes[0].set_title("Boxplot of Final Exam Scores", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Final Examination Score", fontsize=10)

sns.boxplot(y=eda_df["Previous_Scores"], ax=axes[1], color="#68D391", width=0.45)
axes[1].set_title("Boxplot of Internal / Previous Scores", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Previous Scores", fontsize=10)

plt.suptitle("Outlier Analysis of Student Academic Scores", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/fig2_scores_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

# ==============================================================================
# Step 9: Scatter Plot - Study Hours vs Final Examination Scores
# ==============================================================================
plt.figure(figsize=(7.5, 4.5))
sns.regplot(
    x="Hours_Studied",
    y="Exam_Score",
    data=eda_df,
    scatter_kws={"alpha": 0.35, "s": 20, "color": "#3182CE"},
    line_kws={"color": "#E53E3E", "linewidth": 2},
)
r_study_exam = eda_df["Hours_Studied"].corr(eda_df["Exam_Score"])
plt.title(f"Study Hours vs. Final Examination Score (r = {r_study_exam:.2f})", fontsize=12, fontweight="bold")
plt.xlabel("Weekly Study Hours (Hours_Studied)", fontsize=10)
plt.ylabel("Final Examination Score (Exam_Score)", fontsize=10)
plt.tight_layout()
plt.savefig("outputs/fig3_study_hours_scatter.png", dpi=300)
plt.show()

# ==============================================================================
# Step 10 & 11: Correlation Matrix and Heatmap
# ==============================================================================
corr_matrix = eda_df.corr()
print("\nPearson Correlation Matrix:")
print(corr_matrix.round(4))

plt.figure(figsize=(7.2, 5.0))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.6,
    square=True,
)
plt.title("Correlation Heatmap of Academic Performance Variables", fontsize=11.5, fontweight="bold", pad=12)
plt.xticks(rotation=20, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/fig4_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nAll figures saved to 'outputs/' successfully.")
