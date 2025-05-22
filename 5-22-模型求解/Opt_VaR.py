import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# === Step 1: 加载 Excel 数据 ===
df = pd.read_excel("Input_data.xlsx")

# === Step 2: 提取有效变量，清洗非法值 ===
df['A_i'] = df['loan_amnt']
df['P_i'] = 1 - df['estimated_default_prob']
df['r_i'] = df['int_rate'] / 100
df['profit'] = df['A_i'] * (df['r_i'] * (1 - df['P_i']) - df['P_i'])

df = df[(df['P_i'] >= 0) & (df['P_i'] <= 1) & (~df['P_i'].isna())].copy()

# === Step 3: 蒙特卡洛模拟违约情况 ℓ_i(s) ~ Bernoulli(P_i) ===
N = len(df)
S = 1000  # 模拟场景数
np.random.seed(42)
L = np.random.binomial(n=1, p=df['P_i'].values[:, None], size=(N, S))

A_i = df['A_i'].values
profit_i = df['profit'].values

# === Step 4: 构建 VaR 优化模型 ===
B = 1e8        # 总预算
m = 30         # 最多投资个数
M = A_i.sum()  # 大M
beta = 0.95
max_eta = 1e7

model = gp.Model("VaR_Investment")

x = model.addVars(N, vtype=GRB.BINARY, name="x")
z = model.addVars(S, vtype=GRB.BINARY, name="z")
eta = model.addVar(vtype=GRB.CONTINUOUS, name="eta")

# 目标函数：最大化期望收益
model.setObjective(gp.quicksum(x[i] * profit_i[i] for i in range(N)), GRB.MAXIMIZE)

# 预算约束
model.addConstr(gp.quicksum(x[i] * A_i[i] for i in range(N)) <= B, "Budget")

# VaR约束
for s in range(S):
    loss_s = gp.quicksum(x[i] * A_i[i] * L[i, s] for i in range(N))
    model.addConstr(loss_s <= eta + M * z[s], name=f"VaR_scenario_{s}")

# 超损情景不超过 (1 - β)·S
model.addConstr(gp.quicksum(z[s] for s in range(S)) <= (1 - beta) * S, "VaR_confidence")
model.addConstr(eta <= max_eta, "VaR_eta_limit")
model.addConstr(gp.quicksum(x[i] for i in range(N)) <= m, "Top_m")

# === Step 5: 求解 ===
model.setParam("OutputFlag", 1)
model.optimize()

# === Step 6: 输出结果 ===
selected = [i for i in range(N) if x[i].X > 0.5]
selected_ids = df.iloc[selected]['id'].tolist()
print("✅ 选中的借款人 ID：", selected_ids)
print("✅ 最优VaR上界 η* =", eta.X)
print("✅ 投资组合期望收益 =", sum(profit_i[i] for i in selected))
