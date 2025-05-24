import pandas as pd
from gurobipy import Model, GRB

# 1. 读取数据
df = pd.read_excel("Input_data.xlsx")

# 2. 数据预处理
df['A_i'] = df['loan_amnt']
df['r_i'] = df['int_rate'] / 100
df['P_i'] = df['estimated_default_prob']  # 已是还款率
df['profit_i'] = (df['r_i'] - 1) * (1 - df['P_i'])  # 可保留用于分析
df['grade'] = df['grade']

# 3. 清洗数据
df = df[~df[['A_i', 'P_i']].isnull().any(axis=1)]
df = df[
    ~df[['A_i', 'P_i']].apply(
        lambda col: col.isna() | (col == float('inf')) | (col == float('-inf'))
    ).any(axis=1)
]

df = df.reset_index(drop=True)

# 4. 分组与参数
group_dict = df.groupby('grade').groups
N = len(df)

B = 1000000000       # 总预算
R_max = 1_000_0000        # 风险容忍度
m = 10000                 # Top-m 限制
alpha_k = {
    'A': 0.4, 'B': 0.3, 'C': 0.2,
    'D': 0.1, 'E': 0.05, 'F': 0.02, 'G': 0.01
}

# 5. 构建模型
model = Model("MaxExpectedRepayment")

x = model.addVars(N, vtype=GRB.BINARY, name="x")

# ✅ 修改目标函数为期望回款：A_i × P_i
model.setObjective(
    sum(x[i] * df.loc[i, 'A_i'] * df.loc[i, 'P_i'] for i in range(N)),
    GRB.MAXIMIZE
)

# 6. 添加约束
model.addConstr(sum(x[i] * df.loc[i, 'A_i'] for i in range(N)) <= B, "Budget")

for grade, idx_list in group_dict.items():
    if grade in alpha_k:
        model.addConstr(
            sum(x[i] * df.loc[i, 'A_i'] for i in idx_list) <= alpha_k[grade] * B,
            f"Grade_{grade}_Limit"
        )

model.addConstr(
    sum(x[i] * df.loc[i, 'A_i'] * (1 - df.loc[i, 'P_i']) for i in range(N)) <= R_max,
    "ExpectedRisk"
)

model.addConstr(sum(x[i] for i in range(N)) <= m, "TopM")

# 7. 求解
model.optimize()

# 8. 输出结果
selected = [(i, df.loc[i, 'A_i'], df.loc[i, 'grade'], df.loc[i, 'r_i'], df.loc[i, 'P_i']) 
            for i in range(N) if x[i].X > 0.5]

result_df = pd.Input_dataFrame(selected, columns=["Index", "Loan_Amount", "Grade", "Interest_Rate", "Repayment_Prob"])
result_df.to_csv("selected_investments_max_repayment.csv", index=False)

print("✅ 最优期望回款组合已保存至 'selected_investments_max_repayment.csv'. ")
print(result_df)
