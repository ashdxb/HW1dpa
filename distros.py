import numpy as np
from matplotlib import pyplot as plt
from utils import *
import scipy.stats as stats

# Load the data
df = get_pandas()

# print histogram for each feature in the dataset and save independently
for col in df.columns:
    plt.figure(figsize=(10, 6))
    plt.hist(df[col], bins=20)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'{col}_distro.png')

sick = df[df['SepsisLabel'] != 0]
healthy = df[df['SepsisLabel'] == 0]

# drop rows with nan values in HR
sick = sick.dropna(subset=['HR'])
healthy = healthy.dropna(subset=['HR'])

# Plot the distributions of the hr feature for sick and healthy patients
plt.figure(figsize=(10, 6))
plt.hist(sick['HR'], bins=20, alpha=0.5, label='Sick')
plt.hist(healthy['HR'], bins=20, alpha=0.5, label='Healthy')
plt.legend()
plt.title('Distribution of HR')
plt.savefig(f'hr_distro.png')

# create p-value test with 0.05 threshold for the hypothesis that the means of the two distributions are equal
p_value = stats.ttest_ind(sick['HR'], healthy['HR']).pvalue
print(f'p-value: {p_value}')
if(p_value > 0.05):
    print('The means are different')
else:
    print('The means are the same')