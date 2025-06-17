import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib import gridspec

# === Font configuration ===
rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.unicode_minus'] = False

# === Load data ===
file_path = "code/result/result-all.xlsx"
df_sheets = pd.read_excel(file_path, sheet_name=None)
df_primal = df_sheets['Primal_result'].copy()
df_var = df_sheets['VaR_result'].copy()
df_cvar = df_sheets['CVaR_result'].copy()

# === Add model labels ===
df_primal["Model"] = "Primal Model"
df_var["Model"] = "VaR Model"
df_cvar["Model"] = "CVaR Model"

df_all = pd.concat([df_primal, df_var, df_cvar], ignore_index=True)
df_all["Expected_Profit"] = df_all["Loan_Amount"] * df_all["Interest_Rate"] * (1 - df_all["Default_Prob"])

# === Grouping for scatter size ===
df_all["Default_Prob_round"] = df_all["Default_Prob"].round(3)
df_all["Interest_Rate_round"] = df_all["Interest_Rate"].round(3)
grouped = df_all.groupby(["Model", "Default_Prob_round", "Interest_Rate_round"]).size().reset_index(name='Count')

# === Figure 1: Joint distribution scatter plot under three models ===
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
models = ["Primal Model", "VaR Model", "CVaR Model"]
palette = "viridis"

for i, model in enumerate(models):
    ax = fig.add_subplot(gs[0, i])
    subset = grouped[grouped["Model"] == model]

    sns.scatterplot(
        data=subset,
        x="Default_Prob_round",
        y="Interest_Rate_round",
        size="Count",
        hue="Count",
        palette=palette,
        sizes=(20, 200),
        alpha=0.7,
        ax=ax,
        legend=False
    )

    ax.set_title(model, fontsize=14)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.3)
    ax.set_xlabel("Default Probability")
    if i == 0:
        ax.set_ylabel("Interest Rate")
    else:
        ax.set_ylabel("")
    ax.set_aspect('equal')

plt.savefig("code/result/model_comparison_scatter.png", dpi=300, bbox_inches='tight')
plt.show()

# === Figure 2: Expected profit distribution — box plot + strip plot ===
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_all, x="Model", y="Expected_Profit", palette="Set2")
sns.stripplot(data=df_all, x="Model", y="Expected_Profit", color='gray', size=2, alpha=0.3, jitter=True)
plt.title("Distribution of Expected Profit per Loan under Different Models", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Expected Profit", fontsize=12)
plt.ylim(df_all["Expected_Profit"].quantile(0.01), df_all["Expected_Profit"].quantile(0.99))  # Remove outliers
plt.tight_layout()
plt.savefig("code/result/model_comparison_expected_profit_boxplot.png", dpi=300)
plt.show()

# === Figure 3: Loan amount distribution — box plot + strip plot ===
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_all, x="Model", y="Loan_Amount", palette="Set3")
sns.stripplot(data=df_all, x="Model", y="Loan_Amount", color='black', size=2, alpha=0.3, jitter=True)
plt.title("Distribution of Loan Amounts under Different Models", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Loan Amount", fontsize=12)
plt.ylim(df_all["Loan_Amount"].quantile(0.01), df_all["Loan_Amount"].quantile(0.99))  # Remove outliers
plt.tight_layout()
plt.savefig("code/result/model_comparison_loan_amount_boxplot.png", dpi=300)
plt.show()

# === Summary table ===
summary = df_all.groupby("Model").agg(
    Loan_Count=("Loan_Amount", "count"),
    Total_Investment=("Loan_Amount", "sum"),
    Avg_Interest_Rate=("Interest_Rate", "mean"),
    Avg_Default_Prob=("Default_Prob", "mean"),
    Total_Expected_Profit=("Expected_Profit", "sum")
).round(2)

print("📊 Summary analysis of the three models:")
print(summary)
