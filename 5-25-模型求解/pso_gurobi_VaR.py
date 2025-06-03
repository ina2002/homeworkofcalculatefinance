import pandas as pd
import numpy as np
import os
from gurobipy import Model, GRB

# === Step 1: 数据加载与预处理 ===
df = pd.read_excel("code/Input_data.csv")
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

    np.save("code/global_best.npy", global_best)
else:
    print("✅ 发现已有 global_best.npy，直接载入")
    global_best = np.load("code/global_best.npy")

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




# === 保存粒子群优化结果 ===
df_pso = df.iloc[selected_indices].copy()
df_pso['selected'] = 1
df_pso['method'] = 'PSO'
df_pso['eta'] = eta_final
df_pso['profit'] = profit_i[selected_indices]
df_pso.to_csv("code/result_VaR_PSO_selected_loans.csv", index=False)

# === 保存 Gurobi 优化结果 ===
if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
    df_gurobi = df.iloc[selected].copy()
    df_gurobi['selected'] = 1
    df_gurobi['method'] = 'Gurobi'
    df_gurobi['eta'] = eta_val
    df_gurobi['profit'] = profit_i[selected]
    df_gurobi.to_csv("code/result_VaR_result_Gurobi_selected_loans.csv", index=False)



# === 标准化 PSO 输出 CSV ===
df['selected_by_pso'] = 0
df.loc[selected_indices, 'selected_by_pso'] = 1
df['expected_profit'] = df['A_i'] * df['r_i'] * (1 - df['P_i'])

df_pso_output = df.loc[selected_indices, ['id', 'loan_amnt', 'int_rate', 'estimated_default_prob', 'expected_profit']]
summary_pso = pd.DataFrame({
    'id': ['Total'],
    'loan_amnt': [df.loc[selected_indices, 'loan_amnt'].sum()],
    'int_rate': [None],
    'estimated_default_prob': [None],
    'expected_profit': [selected_profits]
})
var_row_pso = pd.DataFrame({
    'id': ['VaR'],
    'loan_amnt': [None],
    'int_rate': [None],
    'estimated_default_prob': [None],
    'expected_profit': [eta_final]
})
df_pso_output = pd.concat([df_pso_output, summary_pso, var_row_pso], ignore_index=True)
df_pso_output.to_csv("code/result_VaR_PSO_selected_loans.csv", index=False)
print("✅ PSO结果已保存至 result_VaR_PSO_selected_loans.csv")

# === 标准化 Gurobi 输出 CSV ===
if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
    df['selected_by_gurobi'] = 0
    df.loc[selected, 'selected_by_gurobi'] = 1

    df_gurobi_output = df.loc[selected, ['id', 'loan_amnt', 'int_rate', 'estimated_default_prob', 'expected_profit']]
    summary_gurobi = pd.DataFrame({
        'id': ['Total'],
        'loan_amnt': [df.loc[selected, 'loan_amnt'].sum()],
        'int_rate': [None],
        'estimated_default_prob': [None],
        'expected_profit': [total_profit]
    })
    var_row_gurobi = pd.DataFrame({
        'id': ['VaR'],
        'loan_amnt': [None],
        'int_rate': [None],
        'estimated_default_prob': [None],
        'expected_profit': [eta_val]
    })
    df_gurobi_output = pd.concat([df_gurobi_output, summary_gurobi, var_row_gurobi], ignore_index=True)
    df_gurobi_output.to_csv("code/result_VaR_Gurobi_selected_loans.csv", index=False)
    print("✅ Gurobi结果已保存至 result_VaR_Gurobi_selected_loans.csv")

    # === 输出摘要 TXT ===
    with open("code/gurobi_VaR_summary.txt", "w") as f:
        f.write("📌 Gurobi 最终求解摘要\n")
        f.write("---------------------------------------------------\n")
        f.write(f"总投资额: {np.sum(x[i].X * A_i[i] for i in range(N)):.2f} / {B:.2f}\n")
        f.write(f"总选择人数: {np.sum(x[i].X for i in range(N)):.0f} / {m}\n")
        f.write(f"投资组合期望收益: {total_profit:.2f}\n")
        f.write(f"VaR 上界 η (95%置信): {eta_val:.2f} / {max_eta:.2f}\n")
    print("✅ Gurobi求解摘要已保存至 gurobi_VaR_summary.txt")
