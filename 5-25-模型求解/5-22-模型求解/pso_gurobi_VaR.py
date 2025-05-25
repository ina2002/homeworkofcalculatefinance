import pandas as pd
import numpy as np
import os
from gurobipy import Model, GRB

# === Step 1: 数据加载与预处理 ===
df = pd.read_excel("5-22-模型求解/Input_data.xlsx")
df['A_i'] = df['loan_amnt']
df['P_i'] = df['estimated_default_prob']
df['r_i'] = df['int_rate'] / 100
df['profit'] = df['A_i'] * (df['r_i'] * (1 - df['P_i']) )
df = df[(df['P_i'] >= 0) & (df['P_i'] <= 1) & (~df['P_i'].isna())].copy()

A_i = df['A_i'].values
P_i = df['P_i'].values
profit_i = df['profit'].values
N = len(df)
S = 1000

np.random.seed(42)
L = np.random.binomial(n=1, p=P_i[:, None], size=(N, S))  # S个场景

# === 参数设置 ===
B = 1e8
m = 5000
beta = 0.95
VaR_alpha = int((1 - beta) * S)
max_eta = 1e7

# === 粒子群参数 ===
pop_size = 30
max_iter = 100
w, c1, c2 = 0.7, 1.5, 1.5

# === PSO初始化 ===
score = profit_i / A_i
sorted_indices = np.argsort(-score)
initial_solution = np.zeros(N, dtype=int)
total_budget = 0
count = 0
for i in sorted_indices:
    if count >= m:
        break
    if total_budget + A_i[i] <= B:
        initial_solution[i] = 1
        total_budget += A_i[i]
        count += 1

positions = np.zeros((pop_size, N), dtype=int)
positions[0] = initial_solution
for p in range(1, pop_size):
    idx = np.random.choice(N, size=m, replace=False)
    while np.sum(A_i[idx]) > B:
        idx = np.random.choice(N, size=m, replace=False)
    positions[p, idx] = 1

velocities = np.random.uniform(-1, 1, size=(pop_size, N))
personal_best = positions.copy()
personal_best_scores = np.full(pop_size, -np.inf)
global_best = None
global_best_score = -np.inf

# === PSO适应度函数 ===
def evaluate(x):
    x = x.astype(int)
    if np.sum(x) > m or np.sum(x * A_i) > B:
        return -1e10
    losses = np.dot((x * A_i), L)
    sorted_losses = np.sort(losses)
    eta = sorted_losses[VaR_alpha]
    if np.sum(losses > eta) > VaR_alpha or eta > max_eta:
        return -1e10
    risk_use_ratio = eta / max_eta
    return np.sum(x * profit_i) - 10000 * risk_use_ratio  # 惩罚项方向修正

# === 判断是否需要重新运行 PSO ===
if not os.path.exists("5-22-模型求解/global_best.npy") or \
    np.load("5-22-模型求解/global_best.npy").shape[0] != N:

    print("⚙️ 正在运行 PSO 寻优...")
    for it in range(max_iter):
        for i in range(pop_size):
            fitness = evaluate(positions[i])
            if fitness > personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best[i] = positions[i].copy()
            if fitness > global_best_score:
                global_best_score = fitness
                global_best = positions[i].copy()
        for i in range(pop_size):
            r1, r2 = np.random.rand(N), np.random.rand(N)
            velocities[i] = (
                w * velocities[i] +
                c1 * r1 * (personal_best[i] - positions[i]) +
                c2 * r2 * (global_best - positions[i])
            )
            sigmoid = 1 / (1 + np.exp(-velocities[i]))
            positions[i] = (np.random.rand(N) < sigmoid).astype(int)

    np.save("5-22-模型求解/global_best.npy", global_best)
else:
    print("✅ 发现已有 global_best.npy，直接载入")
    global_best = np.load("5-22-模型求解/global_best.npy")

# === PSO结果展示 ===
selected_indices = np.where(global_best == 1)[0]
selected_ids = df.iloc[selected_indices]['id'].tolist()
selected_profits = profit_i[selected_indices].sum()
eta_final = np.sort(np.dot((global_best * A_i), L))[VaR_alpha]

print("\n--- 粒子群优化（PSO）结果 ---")
print("✅ 选中借款人数 =", len(selected_indices))
print("✅ PSO 投资组合期望收益 = {:.2f}".format(selected_profits))
print("✅ PSO 使用的 VaR 上界 η = {:.2f}".format(eta_final))

# === Gurobi 求解 ===
# 更稳健的 big_M：所有场景中最大潜在总损失
big_M = np.max((A_i[:, None] * L).sum(axis=0))

model = Model("P2P-VaR-MILP")
model.setParam("OutputFlag", 1)

x = model.addVars(N, vtype=GRB.BINARY, name="x")
eta = model.addVar(vtype=GRB.CONTINUOUS, name="eta")
z = model.addVars(S, vtype=GRB.BINARY, name="z")

model.addConstr(sum(x[i] * A_i[i] for i in range(N)) <= B, name="budget")
model.addConstr(sum(x[i] for i in range(N)) <= m, name="top_m")
for s in range(S):
    loss_expr = sum(x[i] * A_i[i] * L[i, s] for i in range(N))
    model.addConstr(loss_expr - eta <= big_M * z[s], name=f"VaR_s{s}")
model.addConstr(sum(z[s] for s in range(S)) <= VaR_alpha, name="VaR_confidence")

model.setObjective(sum(x[i] * profit_i[i] for i in range(N)), GRB.MAXIMIZE)

# 设置 warm-start
for i in range(N):
    x[i].start = int(global_best[i])

model.optimize()

# === 输出结果 ===
if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
    selected = [i for i in range(N) if x[i].X > 0.5]
    eta_val = eta.X
    total_profit = sum(profit_i[i] for i in selected)
    selected_ids = df.iloc[selected]['id'].tolist()

    print("\n--- Gurobi 精确求解结果 ---")
    print("✅ Gurobi 选中借款人数 =", len(selected))
    print("✅ Gurobi 投资组合期望收益 = {:.2f}".format(total_profit))
    print("✅ Gurobi 计算的 VaR 上界 η = {:.2f}".format(eta_val))
else:
    print("❌ 求解失败, 状态码：", model.status)

