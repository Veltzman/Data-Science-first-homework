import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Finding the exact path of the directory containing this Python file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Joining the directory path with the dataset file name
file_path = os.path.join(current_dir, 'Video_Games_Sales_as_at_22_Dec_2016.csv')

# 3.1 Meta-Analysis: Checking file size
file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"File size: {file_size_mb:.2f} MB\n")

# Loading the data into a DataFrame
df = pd.read_csv(file_path)

# 3.2 Data structure: Number of rows and columns
print(f"Shape of the dataset (Rows, Columns): {df.shape}\n")

# Data types and missing values
print("Data types and missing values:")
df.info()
print("\n")

# 4.1 Missing Values
print("--- Missing Values ---")
missing_data = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage (%)': missing_percent})
print(missing_df[missing_df['Missing Values'] > 0].sort_values(by='Percentage (%)', ascending=False))
print("\n")

# 4.2 Duplicates
print("--- Duplicates ---")
full_duplicates = df.duplicated().sum()
print(f"Full duplicates (exact row match): {full_duplicates}")

# Partial duplicates - e.g., the exact same game on the same platform
partial_duplicates = df.duplicated(subset=['Name', 'Platform']).sum()
print(f"Partial duplicates (same Name and Platform): {partial_duplicates}\n")

# 4.3 Suspicious Values
print("--- Suspicious Values ---")
print("Year of Release bounds:")
print(f"Min Year: {df['Year_of_Release'].min()}, Max Year: {df['Year_of_Release'].max()}")
print("\nChecking for 'tbd' (To Be Determined) in User_Score:")
tbd_count = (df['User_Score'] == 'tbd').sum()
print(f"Number of 'tbd' values: {tbd_count}\n")

# 4.4 Cardinality
print("--- Cardinality ---")
cardinality = df.nunique()
print(cardinality.sort_values())

# --- 5.1 Numeric Variables (Focusing on Global_Sales) ---
print("--- 5.1 Numeric Variables (Global_Sales) ---")
sales = df['Global_Sales'].dropna()

# Central metrics
mean_val = sales.mean()
median_val = sales.median()
std_val = sales.std()
min_val = sales.min()
max_val = sales.max()
q1 = sales.quantile(0.25)
q3 = sales.quantile(0.75)
iqr = q3 - q1
mad_val = (sales - median_val).abs().median()

print(f"Mean: {mean_val:.3f}, Median: {median_val:.3f}, Std: {std_val:.3f}")
print(f"Min: {min_val:.3f}, Max: {max_val:.3f}")
print(f"Q1 (25%): {q1:.3f}, Q3 (75%): {q3:.3f}, IQR: {iqr:.3f}, MAD: {mad_val:.3f}")

# Bias / Skewness
skewness = sales.skew()
print(f"Skewness: {skewness:.3f} (Positive = Right-skewed)")

# Outlier Detection - Method 1: Z-Score
z_scores = (sales - mean_val) / std_val
outliers_z = sales[z_scores.abs() > 3]

# Outlier Detection - Method 2: IQR Rule
outliers_iqr = sales[(sales < (q1 - 1.5 * iqr)) | (sales > (q3 + 1.5 * iqr))]

# Outlier Detection - Method 3: Modified Z-Score (using MAD)
# Formula: 0.6745 * (x - median) / MAD
mod_z_scores = 0.6745 * (sales - median_val) / mad_val
outliers_mad = sales[mod_z_scores.abs() > 3.5]

print(f"\nOutliers detected by Z-Score: {len(outliers_z)}")
print(f"Outliers detected by IQR: {len(outliers_iqr)}")
print(f"Outliers detected by Modified Z-Score (MAD): {len(outliers_mad)}\n")


# --- 5.2 Categorical Variables (Focusing on Genre) ---
print("--- 5.2 Categorical Variables (Genre) ---")
genre = df['Genre'].dropna()
total_count = genre.count()

# Frequencies and Mode
frequencies = genre.value_counts()
mode_val = frequencies.index[0]
mode_pct = (frequencies.iloc[0] / total_count) * 100
print(f"Mode (Most frequent): {mode_val} ({mode_pct:.2f}% of data)")

# Top K (Let's choose K=3)
k = 3
top_k = frequencies.head(k)
top_k_pct = (top_k.sum() / total_count) * 100
print(f"Top {k} genres hold {top_k_pct:.2f}% of the data.")

# Minimum values for P% (Let's choose P=80%)
p = 80
cumulative_pct = (frequencies / total_count * 100).cumsum()
min_values_p = cumulative_pct[cumulative_pct <= p].index.tolist()
if len(min_values_p) < len(frequencies):
    min_values_p.append(cumulative_pct[cumulative_pct > p].index[0])
    
print(f"Minimum categories to represent {p}% of data: {len(min_values_p)} categories")
print(f"These categories are: {min_values_p}")

# Rare categories (less than 2% of the data)
rare_categories = frequencies[frequencies / total_count < 0.02]
print(f"Rare categories (under 2%): {rare_categories.index.tolist()}")



# (Assuming df is already loaded from the previous steps)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(current_dir, 'Video_Games_Sales_as_at_22_Dec_2016.csv')
# df = pd.read_csv(file_path)

# Pre-processing: Convert User_Score to numeric, turning 'tbd' into NaN
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')

# --- 6.1 Numeric-Numeric Correlations ---
print("--- 6.1 Numeric Correlations ---")
numeric_cols = ['Critic_Score', 'User_Score', 'Global_Sales', 'Year_of_Release']
num_df = df[numeric_cols].dropna()

print("Pearson Correlation:\n", num_df.corr(method='pearson'), "\n")
print("Spearman Correlation:\n", num_df.corr(method='spearman'), "\n")
print("Kendall Correlation:\n", num_df.corr(method='kendall'), "\n")

# --- 6.2 Categorical-Categorical & Categorical-Numeric (Binning) ---
print("--- 6.2 Binning and Relationships ---")
# Binning Global_Sales into categories (Low, Medium, High)
df['Sales_Category'] = pd.qcut(df['Global_Sales'], q=3, labels=['Low', 'Medium', 'High'])
crosstab_genre_sales = pd.crosstab(df['Genre'], df['Sales_Category'])
print("Cross-tabulation (Genre vs Sales Category):\n", crosstab_genre_sales, "\n")

# --- 6.3 Graphs ---
# Setting up the visual style
sns.set_theme(style="whitegrid")

# 1. Scatterplot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Critic_Score', y='Global_Sales', alpha=0.5)
plt.title('Scatterplot: Critic Score vs Global Sales')
plt.xlabel('Critic Score')
plt.ylabel('Global Sales (Millions)')
plt.show()

# 2. Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['Critic_Score'].dropna(), bins=20, kde=True)
plt.title('Histogram: Distribution of Critic Scores')
plt.xlabel('Critic Score')
plt.ylabel('Frequency')
plt.show()

# 3. Barchart
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='Genre', order=df['Genre'].value_counts().index)
plt.title('Barchart: Number of Games per Genre')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# 4. Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Genre', y='Critic_Score')
plt.title('Boxplot: Critic Scores across Genres')
plt.xticks(rotation=45)
plt.xlabel('Genre')
plt.ylabel('Critic Score')
plt.show()

# 5. Violin Plot (Filtered to sales < 2M for better visibility)
plt.figure(figsize=(12, 6))
sns.violinplot(data=df[df['Global_Sales'] < 2], x='Genre', y='Global_Sales')
plt.title('Violin Plot: Global Sales (< 2M) by Genre')
plt.xticks(rotation=45)
plt.xlabel('Genre')
plt.ylabel('Global Sales')
plt.show()

# 6. Pie Chart
plt.figure(figsize=(8, 8))
platform_counts = df['Platform'].value_counts().head(5)
plt.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart: Top 5 Platforms by Number of Releases')
plt.show()

# 7. Pairplot
sns.pairplot(num_df.sample(500)) # Sampling 500 rows for performance
plt.suptitle('Pairplot of Numeric Variables', y=1.02)
plt.show()

# 8. Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(num_df.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap: Spearman Correlation Matrix')
plt.show()

# --- 7. Index Analysis ---
print("--- 7. Index Analysis ---")

# Is the index unique?
is_unique = df.index.is_unique
print(f"Is the index unique? {is_unique}")

# Is the index time-based?
is_time_based = isinstance(df.index, pd.DatetimeIndex)
print(f"Is the index time-based (DatetimeIndex)? {is_time_based}")

# Is the index sorted?
is_sorted = df.index.is_monotonic_increasing
print(f"Is the index sorted (monotonic increasing)? {is_sorted}")


print("--- 9. Bonus: Engineering and Hypothesis Testing ---")

# Feature Engineering 1: Age of the game (Assuming current year is 2016 for dataset context)
df['Game_Age_Years'] = 2016 - df['Year_of_Release']

# Feature Engineering 2: Binary target for Classification models (Is it a Hit?)
# Let's define a "Hit" as selling more than 1 Million copies globally
df['Is_Hit'] = (df['Global_Sales'] > 1.0).astype(int)

print("New Engineered Features preview:")
print(df[['Name', 'Year_of_Release', 'Game_Age_Years', 'Global_Sales', 'Is_Hit']].head())
print("\n")

# Hypothesis Testing: Independent T-Test
# Hypothesis: Do 'Action' games sell significantly different amounts globally compared to 'Role-Playing' games?
action_sales = df[df['Genre'] == 'Action']['Global_Sales'].dropna()
rpg_sales = df[df['Genre'] == 'Role-Playing']['Global_Sales'].dropna()

t_stat, p_value = stats.ttest_ind(action_sales, rpg_sales, equal_var=False)

print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_value:.4e}")
if p_value < 0.05:
    print("Conclusion: Reject the null hypothesis. There is a statistically significant difference in mean sales.")
else:
    print("Conclusion: Fail to reject the null hypothesis. No significant difference in mean sales.")