import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

df = pd.read_excel("Input_data.xlsx")

df['A_i'] = df['loan_amnt']
df['P_i'] = 1 - df['estimated_default_prob']
df['r_i'] = df['int_rate'] / 100
df['profit'] = df['A_i'] * (df['r_i'] * (1 - df['P_i']) - df['P_i'])
df = df[(df['P_i'] >= 0) & (df['P_i'] <= 1) & (~df['P_i'].isna())].copy()

N = len(df)
S = 1000
np.random.seed(42)
L = np.random.binomial(n=1, p=df['P_i'].values[:, None], size=(N, S))

A_i = df['A_i'].values
profit_i = df['profit'].values

B = 1e8
m = 30
beta = 0.95
M = A_i.sum()
max_eta = 1e5

model = gp.Model("VaR_Investment")

x = model.addVars(N, vtype=GRB.BINARY, name="x")
z = model.addVars(S, vtype=GRB.BINARY, name="z")
eta = model.addVar(vtype=GRB.CONTINUOUS, name="eta")

model.setObjective(gp.quicksum(x[i] * profit_i[i] for i in range(N)), GRB.MAXIMIZE)

model.addConstr(gp.quicksum(x[i] * A_i[i] for i in range(N)) <= B, "Budget")

for s in range(S):
    loss_s = gp.quicksum(x[i] * A_i[i] * L[i, s] for i in range(N))
    model.addConstr(loss_s <= eta + M * z[s], name=f"VaR_scenario_{s}")

model.addConstr(gp.quicksum(z[s] for s in range(S)) <= (1 - beta) * S, "VaR_confidence")
model.addConstr(eta <= max_eta, "VaR_eta_limit")
model.addConstr(gp.quicksum(x[i] for i in range(N)) <= m, "Top_m")

model.setParam("OutputFlag", 1)
model.optimize()

selected = [i for i in range(N) if x[i].X > 0.5]
selected_ids = df.iloc[selected]['id'].tolist()
print("✅ Gurobi 选中借款人 ID：", selected_ids)
print("✅ 最优 VaR 上界 η =", eta.X)
print("✅ 投资组合期望收益 =", sum(profit_i[i] for i in selected))