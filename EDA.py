## Import packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

## Displa full ouptut in pycharm console
pd.set_option('display.max_rows', None)            # Display all rows
pd.set_option('display.max_columns', None)         # Display all columns
pd.set_option('display.width', None)               # No truncation of content
pd.set_option('display.expand_frame_repr', False)  # Don't wrap DataFrame display

## Load the dataset
cwd = os.getcwd()
df = pd.read_csv(cwd + f"\\Charging_events_data - charging_events_meter_reading.csv")
df.head()

## Data cleaning and formatting
df.info()
df['Start Time'] = pd.to_datetime(df['Start Time'], format='%d.%m.%Y %H:%M')
df[['Meter Start (Wh)', 'Total Duration (s)']] = df[['Meter Start (Wh)', 'Total Duration (s)']].astype(float)
df.info()

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())
print("Proportion of missing values in each column:")
print(df.isnull().sum()/len(df) * 100)

df['Charger_name'] = df['Charger_name'].fillna('unknown') # Replace missing charger_name values with 'unknown'

df.to_csv("EDA.csv", index=False)

## EDA

## Descriptive statistics
print("Descriptive statistics:")
print(df.describe())

## Create histograms for 'Meter Total(Wh)' and 'Total Duration (s)'
plt.figure(figsize = (10, 5))

# Histogram for 'Meter Total(Wh)'
plt.subplot(1, 2, 1)
sns.histplot(df['Meter Total(Wh)'], kde = True)
plt.title("Histogram of Meter Total(Wh)")

# Histogram for 'Total Duration (s)'
plt.subplot(1, 2, 2)
sns.histplot(df['Total Duration (s)'], kde = True)
plt.title("Histogram of Total Duration (s)")

plt.tight_layout()  # Adjust spacing to prevent overlap
overall_plot_path = os.path.join("Plots", "overall_histogram.png")
plt.savefig(overall_plot_path)
plt.show()

## Temporal analysis

### Time Trends: Peak periods for charging events
# Extract time components
df['Hour'] = df['Start Time'].dt.hour
df['Day of Week'] = df['Start Time'].dt.dayofweek  # Monday=0, Sunday=6
df['ay of Week'] = (df['Day of Week'] + 1) % 7 # Adjust to map Sunday as 0 and Saturday as 6
df['Month'] = df['Start Time'].dt.month  # January=1, December=12

# Adjust day of week mapping to start with Sunday
day_of_week_mapping = {
    0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed",
    4: "Thu", 5: "Fri", 6: "Sat"
}
month_mapping = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# Apply the mappings to convert numeric days and months to strings
df['Day of Week'] = df['Day of Week'].map(day_of_week_mapping)
# Create a new DataFrame with the desired order for plotting
day_order_df = pd.Categorical(df['Day of Week'], categories = ["Sun", "Mon", "Tue",
                                                               "Wed", "Thu",
                                                               "Fri", "Sat"], ordered = True)
df['Ordered Day of Week'] = day_order_df

# Create a 'Year-Month' column
df['Year-Month'] = df['Start Time'].dt.to_period('M').astype(str)
unique_periods = df['Year-Month'].unique()

# Convert these unique periods to Pandas Period and sort them chronologically
sorted_periods = sorted(
    [pd.Period(period, freq='M') for period in unique_periods]
)

# Ensure correct order by converting sorted periods back to strings
ordered_categories = [str(period) for period in sorted_periods]

# Create a Categorical type with the correct chronological order
df['Year-Month'] = pd.Categorical(df['Year-Month'], categories=ordered_categories, ordered=True)

# Create histograms with specified orders
plt.figure(figsize=(15, 5))

# Histogram for hours
plt.subplot(1, 2, 1)
sns.histplot(df['Hour'], kde = True)
plt.title("Distribution of Charging Events by Hour")

# Histogram for days of the week with specified order
plt.subplot(1, 2, 2)
sns.histplot(df['Ordered Day of Week'], kde = True)
plt.title("Distribution of Charging Events by Day of Week")

plt.tight_layout()  # Adjust spacing
overall_plot_path = os.path.join("Plots", "event_by_period.png")
plt.savefig(overall_plot_path)
plt.show()

# Histogram for 'Year-Month'
sns.histplot(df['Year-Month'], kde=True)
plt.title("Distribution of Charging Events by Month")
plt.xlabel("Month")
overall_plot_path = os.path.join("Plots", "event_by_month.png")
plt.savefig(overall_plot_path)
plt.show()

# Investigate outliers in duration
outlier_threshold = df['Total Duration (s)'].quantile(0.95)  # Top 5% as outliers
outliers = df[df['Total Duration (s)'] > outlier_threshold]
outliers_prop = round(len(outliers)/len(df) * 100, 1)

print("Outliers in Total Duration (s):")
print(outliers)
print(f"Relative to the general duration of charging events, {outliers_prop}% (i.e.",
      f", {len(outliers)}) of the observations can be considered outliers.")

### Temporal Correlations
# Calculate correlation between 'Start Time' and 'Total Duration (s)'
df['Start Timestamp'] = df['Start Time'].astype('int64') // 10**9  # Convert to Unix timestamp
correlation = df[['Start Timestamp', 'Total Duration (s)']].corr()

print("Correlation between Start Time and Total Duration (s):")
print(correlation)

## Calculate correlations between numerical columns
correlation_matrix = df[["Meter Start (Wh)", "Meter End(Wh)", "Meter Total(Wh)", "Total Duration (s)"]].corr()
# Plot a heatmap to visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix for Numerical Columns")
overall_plot_path = os.path.join("Plots", "correlation_between_numerical variables.png")
plt.savefig(overall_plot_path)
plt.show()

## Plot a bar plot for the mean 'Meter Total(Wh)' for each charger
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Charger_name", y="Meter Total(Wh)", ci="sd")
plt.title("Mean Meter Total(Wh) by Charger")
overall_plot_path = os.path.join("Plots", "Meter Total(Wh) by charger.png")
plt.savefig(overall_plot_path)
plt.show()

## Whatts vs Seconds scatterplot by charger
plt.figure(figsize = (8, 6))
sns.scatterplot(data = df, x = "Meter Total(Wh)", y = "Total Duration (s)",
                hue = "Charger_name", palette = 'tab10', alpha = 0.7)
plt.title("Meter Total(Wh) vs. Total Duration (s)")
plt.xlabel("Meter Total(Wh)")
plt.ylabel("Total Duration (s)")
overall_plot_path = os.path.join("Plots", "Whatts vs Seconds scatterplot.png")
plt.savefig(overall_plot_path)
plt.show()