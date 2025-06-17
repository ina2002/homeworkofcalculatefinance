import pandas as pd
import numpy as np
import os
from gurobipy import Model, GRB

# === Step 1: 数据加载与预处理 ===
df = pd.read_excel("code/Input_data.csv")
df = df[(~df['estimated_default_prob'].isna()) & 
        (df['estimated_default_prob'] >= 0) & 
        (df['estimated_default_prob'] <= 1)].copy()
df['A_i'] = df['loan_amnt']
df['P_i'] = df['estimated_default_prob']
df['r_i'] = df['int_rate'] / 100
df['profit'] = df['A_i'] * df['r_i'] * (1 - df['P_i'])

A_i = df['A_i'].values
P_i = df['P_i'].values
profit_i = df['profit'].values
N = len(df)
S = 1000

# === Step 2: 蒙特卡洛模拟 ===
np.random.seed(42)
L = np.random.binomial(n=1, p=P_i[:, None], size=(N, S))

# === Step 3: 参数设置 ===
B = 1e8
m = 5000
beta = 0.95
VaR_alpha = int((1 - beta) * S)
max_eta = 1e7

group_dict = df.groupby('grade').groups
N = len(df)


alpha_k = {         # 信用等级投资比例上限
    'A': 0.4, 'B': 0.3, 'C': 0.2,
    'D': 0.1, 'E': 0.05, 'F': 0.02, 'G': 0.01
}
# === Step 4: PSO 参数与初始化 ===
pop_size = 30
max_iter = 100
w, c1, c2 = 0.7, 1.5, 1.5

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

# === Step 5: PSO适应度函数 ===
def evaluate(x):
    x = x.astype(int)

    # 基本约束：人数、预算
    if np.sum(x) > m or np.sum(x * A_i) > B:
        return -1e10

    # 信用等级比例约束
    for grade, idx_list in group_dict.items():
        if grade in alpha_k:
            total_grade_amount = sum(A_i[i] * x[i] for i in idx_list)
            if total_grade_amount > alpha_k[grade] * B:
                return -1e10

    # 期望损失约束（新增）
    expected_loss = np.sum(x * A_i * P_i)
    if expected_loss > 15000000:
        return -1e10

    # VaR约束
    losses = np.dot((x * A_i), L)
    sorted_losses = np.sort(losses)
    eta = sorted_losses[VaR_alpha]
    if np.sum(losses > eta) > VaR_alpha or eta > max_eta:
        return -1e10

    # 返回带惩罚项的收益（目标最大化）
    return np.sum(x * profit_i) - 10000 * (eta / max_eta)


# === Step 6: 运行 PSO ===
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

# === Step 7: PSO输出 ===
selected_indices = np.where(global_best == 1)[0]
selected_profits = profit_i[selected_indices].sum()
eta_final = np.sort(np.dot((global_best * A_i), L))[VaR_alpha]

print("\n--- PSO结果 ---")
print(f"✅ 选中借款人数 = {len(selected_indices)}")
print(f"✅ 投资组合期望收益 = {selected_profits:.2f}")
print(f"✅ 使用的 VaR 上界 η = {eta_final:.2f}")

# === Step 8: Gurobi 求解 ===
print("\n⚙️ 正在运行 Gurobi 求解...")
big_M = np.max((A_i[:, None] * L).sum(axis=0))

model = Model("P2P-VaR-MILP")
model.setParam("OutputFlag", 0)

x = model.addVars(N, vtype=GRB.BINARY, name="x")
eta = model.addVar(vtype=GRB.CONTINUOUS, name="eta")
z = model.addVars(S, vtype=GRB.BINARY, name="z")
# 信用等级比例约束
for grade, idx_list in group_dict.items():
    if grade in alpha_k:
        model.addConstr(
            sum(x[i] * df.loc[i, 'A_i'] for i in idx_list) <= alpha_k[grade] * B,
            f"Grade_{grade}_Limit"
        )
model.addConstr(sum(x[i] * A_i[i] for i in range(N)) <= B)
model.addConstr(sum(x[i] for i in range(N)) <= m)
for s in range(S):
    loss_expr = sum(x[i] * A_i[i] * L[i, s] for i in range(N))
    model.addConstr(loss_expr - eta <= big_M * z[s])
model.addConstr(sum(z[s] for s in range(S)) <= VaR_alpha)
# 风险约束：期望损失（A_i × P_i）不超过 R_max
model.addConstr(
    sum(x[i] * df.loc[i, 'A_i'] * df.loc[i, 'P_i'] for i in range(N)) <= 15000000,
    "ExpectedRisk"
)

model.setObjective(sum(x[i] * profit_i[i] for i in range(N)), GRB.MAXIMIZE)

for i in range(N):
    x[i].start = int(global_best[i])

model.optimize()

# === Step 9: Gurobi输出整理 ===
if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
    x_array = np.array([x[i].X for i in range(N)])
    selected_gurobi = np.where(x_array > 0.5)[0]
    profit_gurobi = profit_i[selected_gurobi].sum()
    eta_val = eta.X

    print("\n--- Gurobi 精确求解结果 ---")
    print(f"✅ 选中借款人数 = {len(selected_gurobi)}")
    print(f"✅ 投资组合期望收益 = {profit_gurobi:.2f}")
    print(f"✅ Gurobi 计算的 VaR 上界 η = {eta_val:.2f}")

    # === 保存 CSV：PSO ===
    df_pso_output = df.loc[selected_indices, ['id', 'loan_amnt', 'int_rate', 'estimated_default_prob']].copy()
    df_pso_output['expected_profit'] = df_pso_output['loan_amnt'] * df_pso_output['int_rate'] / 100 * (1 - df_pso_output['estimated_default_prob'])
    summary_pso = pd.DataFrame({'id': ['Total'], 'loan_amnt': [df_pso_output['loan_amnt'].sum()],
                                'int_rate': [None], 'estimated_default_prob': [None],
                                'expected_profit': [selected_profits]})
    var_row_pso = pd.DataFrame({'id': ['VaR'], 'loan_amnt': [None], 'int_rate': [None],
                                'estimated_default_prob': [None], 'expected_profit': [eta_final]})
    df_pso_output = pd.concat([df_pso_output, summary_pso, var_row_pso], ignore_index=True)
    df_pso_output.to_csv("code/VaR_PSO_result.csv", index=False)

    # === 保存 CSV：Gurobi ===
    df_gurobi_output = df.loc[selected_gurobi, ['id', 'loan_amnt', 'int_rate', 'estimated_default_prob']].copy()
    df_gurobi_output['expected_profit'] = df_gurobi_output['loan_amnt'] * df_gurobi_output['int_rate'] / 100 * (1 - df_gurobi_output['estimated_default_prob'])
    summary_gurobi = pd.DataFrame({'id': ['Total'], 'loan_amnt': [df_gurobi_output['loan_amnt'].sum()],
                                   'int_rate': [None], 'estimated_default_prob': [None],
                                   'expected_profit': [profit_gurobi]})
    var_row_gurobi = pd.DataFrame({'id': ['VaR'], 'loan_amnt': [None], 'int_rate': [None],
                                   'estimated_default_prob': [None], 'expected_profit': [eta_val]})
    df_gurobi_output = pd.concat([df_gurobi_output, summary_gurobi, var_row_gurobi], ignore_index=True)
    df_gurobi_output.to_csv("code/VaR_Gurobi_result.csv", index=False)

    # === 保存 TXT 摘要 ===
    with open("code/VaR_summary.txt", "w") as f:
        f.write("Gurobi 最终求解摘要\n")
        f.write("---------------------------------------------------\n")
        f.write(f"总投资额: {np.sum(x_array * A_i):.2f} / {B:.2f}\n")
        f.write(f"总选择人数: {np.sum(x_array):.0f} / {m}\n")
        f.write(f"投资组合期望收益: {profit_gurobi:.2f}\n")
        f.write(f"VaR 上界 η: {eta_val:.2f} / {max_eta:.2f}\n")
    print("✅ 输出结果已保存至 CSV 与 TXT")
else:
    print("❌ Gurobi 求解失败，状态码：", model.status)
