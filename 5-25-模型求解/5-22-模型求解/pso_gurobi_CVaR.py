import pandas as pd
import numpy as np

# === Step 1: 数据加载与预处理 ===
df = pd.read_excel("Input_data.xlsx", sheet_name="Sheet1")
df = df[df['estimated_default_prob'].notna()].copy()
df['P_i'] = ( df['estimated_default_prob']).clip(0, 1)
df['A_i'] = df['loan_amnt']
df['r_i'] = df['int_rate'] / 100
df['profit'] = df['A_i'] * df['r_i'] * (1 - df['P_i']) 

A_i = df['A_i'].values
P_i = df['P_i'].values
profit_i = df['profit'].values
N = len(df)

# === Step 2: 蒙特卡洛模拟损失矩阵 ===
S = 1000
np.random.seed(42)
L = np.random.binomial(n=1, p=P_i[:, None], size=(N, S)) * A_i[:, None]

# === Step 3: 参数设置 ===
B = 1e8
m = 30
beta = 0.95
R_max = 1.5e7
inv_tail = 1 / ((1 - beta) * S)

# === Step 4: PSO 参数 ===
pop_size = 30
max_iter = 100
w = 0.7
c1 = 1.5
c2 = 1.5

# === Step 5: 初始化粒子（启发式） ===
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

# === Step 6: CVaR适应度函数 ===
def evaluate(x):
    x = x.astype(int)
    if np.sum(x) > m or np.sum(x * A_i) > B:
        return -1e10
    loss_s = np.dot((x * A_i), L)
    eta = np.percentile(loss_s, beta * 100)
    xi = np.maximum(loss_s - eta, 0)
    cvar = eta + inv_tail * np.sum(xi)
    if cvar > R_max:
        return -1e10
    # ✅ 奖励靠近 R_max（使用更多风险）
    risk_util_ratio = cvar / R_max
    return np.sum(x * profit_i) + 10000 * risk_util_ratio


# === Step 7: PSO 主循环 ===
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

# === Step 8: 输出结果 ===
selected_indices = np.where(global_best == 1)[0]
selected_ids = df.iloc[selected_indices]['id'].tolist()
selected_profits = profit_i[selected_indices].sum()
loss_s = np.dot((global_best * A_i), L)
eta = np.percentile(loss_s, beta * 100)
xi = np.maximum(loss_s - eta, 0)
cvar = eta + inv_tail * np.sum(xi)

print("✅ PSO-CVaR 选中借款人 ID：", selected_ids)
print("✅ PSO-CVaR 投资组合期望收益 =", selected_profits)
print("✅ PSO-CVaR 风险值 CVaR =", cvar)


from gurobipy import Model, GRB

# === Step 9: 用 Gurobi 进一步优化 ===
model = Model("CVaR_Optimization")
model.setParam("OutputFlag", 0)

x = model.addVars(N, vtype=GRB.BINARY, name="x")
eta = model.addVar(lb=0, name="eta")
xi = model.addVars(S, lb=0, name="xi")

# 目标函数：期望收益 + 风险利用率激励
model.setObjective(
    sum(x[i] * profit_i[i] for i in range(N)) + 10000 * eta / R_max, GRB.MAXIMIZE
)

# 预算约束
model.addConstr(sum(x[i] * A_i[i] for i in range(N)) <= B)

# 个数约束
model.addConstr(sum(x[i] for i in range(N)) <= m)

# CVaR约束
for s in range(S):
    loss_s = sum(x[i] * L[i, s] for i in range(N))
    model.addConstr(xi[s] >= loss_s - eta)

model.addConstr(
    eta + inv_tail * sum(xi[s] for s in range(S)) <= R_max
)

# 初始解
for i in range(N):
    x[i].Start = int(global_best[i])

model.optimize()

# === Step 10: 结果提取 ===
selected_ids_gurobi = [df.iloc[i]['id'] for i in range(N) if x[i].X > 0.5]
profit_gurobi = sum(profit_i[i] for i in range(N) if x[i].X > 0.5)
loss_vector = np.dot(
    np.array([x[i].X for i in range(N)]) * A_i, L
)
eta_val = np.percentile(loss_vector, beta * 100)
xi_val = np.maximum(loss_vector - eta_val, 0)
cvar_val = eta_val + inv_tail * np.sum(xi_val)

print("📌 Gurobi 精调选中借款人 ID：", selected_ids_gurobi)
print("📌 Gurobi 投资组合期望收益 =", profit_gurobi)
print("📌 Gurobi 风险值 CVaR =", cvar_val)
