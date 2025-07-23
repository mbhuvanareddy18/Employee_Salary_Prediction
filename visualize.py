import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/employee_data.csv")

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Salary by Job Title
plt.figure(figsize=(8,5))
sns.barplot(x="Job_Title", y="Salary", data=df)
plt.xticks(rotation=45)
plt.title("Salary Distribution by Job Title")
plt.show()
