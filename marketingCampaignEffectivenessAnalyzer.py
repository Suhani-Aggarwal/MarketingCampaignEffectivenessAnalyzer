import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# LOAD DATASET
df = pd.read_csv("marketing_campaign_dataset.csv")

print(df.head())
print(df.shape)
print(df.info())

# DATA PREPROCESSING
print(df.isnull().sum())

df = df.drop_duplicates()

df["Date"] = pd.to_datetime(df["Date"])

df["Acquisition_Cost"] = df["Acquisition_Cost"].replace('[\$,]', '', regex=True).astype(float)

# CREATE NEW METRICS
df["CTR"] = df["Clicks"] / df["Impressions"].replace(0,1)
df["CPC"] = df["Acquisition_Cost"] / df["Clicks"].replace(0,1)

print(df[["CTR","CPC"]].head())

# 4. CAMPAIGN TYPE ANALYSIS
campaign_conv = df.groupby("Campaign_Type")["Conversion_Rate"].mean()

plt.figure(figsize=(8,5))
sns.barplot(x=campaign_conv.index, y=campaign_conv.values)
plt.xticks(rotation=45)
plt.title("Average Conversion Rate by Campaign Type")
plt.xlabel("Campaign Type")
plt.ylabel("Conversion Rate")

plt.savefig("images/campaign_conversion_rate.png")

plt.show()


# 5. CHANNEL PERFORMANCE
channel_roi = df.groupby("Channel_Used")["ROI"].mean()

plt.figure(figsize=(8,5))
channel_roi.plot(kind="bar")
plt.title("Average ROI by Marketing Channel")
plt.xlabel("Channel Used")
plt.ylabel("ROI")
plt.xticks(rotation=45)

plt.savefig("images/channel_roi.png")

plt.show()


# 6. CUSTOMER SEGMENT ANALYSIS
segment = df.groupby("Customer_Segment")["Conversion_Rate"].mean()

plt.figure(figsize=(8,5))
segment.plot(kind="bar")
plt.title("Conversion Rate by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Conversion Rate")
plt.xticks(rotation=45)

plt.savefig("images/customer_segment_conversion.png")

plt.show()


# 7. ENGAGEMENT VS CONVERSION
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="Engagement_Score",
    y="Conversion_Rate",
    data=df
)

plt.title("Engagement vs Conversion")
plt.xlabel("Engagement Score")
plt.ylabel("Conversion Rate")

plt.savefig("images/engagement_vs_conversion.png")

plt.show()


# 8. MONTHLY TREND ANALYSIS
monthly = df.groupby(df["Date"].dt.month)["Conversion_Rate"].mean()

plt.figure(figsize=(8,5))
monthly.plot(marker="o")
plt.title("Monthly Conversion Trend")
plt.xlabel("Month")
plt.ylabel("Conversion Rate")

plt.savefig("images/monthly_conversion_trend.png")

plt.show()


# 9. A/B TESTING (EMAIL VS FACEBOOK)
email = df[df["Channel_Used"]=="Email"]["Conversion_Rate"]
facebook = df[df["Channel_Used"]=="Facebook"]["Conversion_Rate"]

print("Email Sample Size:", len(email))
print("Facebook Sample Size:", len(facebook))

t,p = ttest_ind(email, facebook)

print("T-test:", t)
print("P-value:", p)

if p < 0.05:
    print("Result: Significant difference between campaigns")
else:
    print("Result: No significant difference")


# 10. MULTIPLE A/B TESTS BETWEEN ALL CHANNELS
print("\nMultiple A/B Testing Between Channels\n")

channels = df["Channel_Used"].unique()

for i in range(len(channels)):
    for j in range(i+1, len(channels)):
        
        group1 = df[df["Channel_Used"] == channels[i]]["Conversion_Rate"]
        group2 = df[df["Channel_Used"] == channels[j]]["Conversion_Rate"]

        t,p = ttest_ind(group1, group2)

        print(f"{channels[i]} vs {channels[j]}")
        print("T-stat:", t)
        print("P-value:", p)

        if p < 0.05:
            print("Significant Difference\n")
        else:
            print("No Significant Difference\n")


# 11. CONVERSION RATE DISTRIBUTION
plt.figure(figsize=(10,6))

sns.boxplot(
    x="Channel_Used",
    y="Conversion_Rate",
    data=df
)

plt.title("Conversion Rate Distribution by Marketing Channel")
plt.xlabel("Channel Used")
plt.ylabel("Conversion Rate")
plt.xticks(rotation=45)

plt.savefig("images/conversion_rate_distribution.png")

plt.show()


# 12. BUSINESS INSIGHTS
print("\nKey Business Insights")

best_channel = df.groupby("Channel_Used")["Conversion_Rate"].mean().idxmax()
best_segment = df.groupby("Customer_Segment")["Conversion_Rate"].mean().idxmax()
best_roi = df.groupby("Channel_Used")["ROI"].mean().idxmax()

print("Best Performing Marketing Channel:", best_channel)
print("Best Customer Segment:", best_segment)
print("Highest ROI Channel:", best_roi)
