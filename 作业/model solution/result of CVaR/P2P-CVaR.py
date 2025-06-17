import pandas as pd
import numpy as np
from gurobipy import Model, GRB

# === Step 1: 数据加载与预处理 ===
df = pd.read_excel("code/Input_data.csv", sheet_name="Sheet1")
df = df[df['estimated_default_prob'].notna()].copy()
df['P_i'] = df['estimated_default_prob'].clip(0, 1)
df['A_i'] = df['loan_amnt']
df['r_i'] = df['int_rate'] / 100
df['profit'] = df['A_i'] * df['r_i'] * (1 - df['P_i'])

A_i = df['A_i'].values
P_i = df['P_i'].values
profit_i = df['profit'].values
N = len(df)

# === Step 2: 蒙特卡洛模拟损失矩阵（只含0/1违约，不乘金额）===
S = 1000
np.random.seed(42)
L = np.random.binomial(n=1, p=P_i[:, None], size=(N, S))  # shape = (N, S)

# === Step 3: 参数设置 ===
B = 1e8
m = 5000
beta = 0.95
R_max = 1.5e7
inv_tail = 1 / ((1 - beta) * S)

# 4. 分组与参数
group_dict = df.groupby('grade').groups
N = len(df)


alpha_k = {         # 信用等级投资比例上限
    'A': 0.4, 'B': 0.3, 'C': 0.2,
    'D': 0.1, 'E': 0.05, 'F': 0.02, 'G': 0.01
}

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

# === Step 6: CVaR适应度函数（修正损失计算） ===
def evaluate(x):
    x = x.astype(int)

    # 人数和预算约束
    if np.sum(x) > m or np.sum(x * A_i) > B:
        return -1e10

    # 新增：期望损失约束
    expected_loss = np.sum(x * A_i * P_i)
    if expected_loss > 15000000:
        return -1e10

    # 计算 CVaR
    loss_s = (L.T @ (x * A_i)).flatten()
    eta = np.percentile(loss_s, beta * 100)
    xi = np.maximum(loss_s - eta, 0)
    cvar = eta + inv_tail * np.sum(xi)

    # CVaR 约束
    if cvar > R_max:
        return -1e10

    # 目标函数：期望收益 + 风险利用效率项
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

# === Step 8: 输出 PSO 结果 ===
print("⚙️ 正在运行 PSO 寻优...")
selected_indices = np.where(global_best == 1)[0]
selected_profits = profit_i[selected_indices].sum()

loss_s = (L.T @ (global_best * A_i)).flatten()
eta = np.percentile(loss_s, beta * 100)
xi = np.maximum(loss_s - eta, 0)
cvar = eta + inv_tail * np.sum(xi)
print("\n --- 粒子群优化（PSO）结果 ---")
print("✅ PSO-CVaR 投资组合期望收益 =", selected_profits)
print("✅ PSO-CVaR 风险值 CVaR =", cvar)
print("✅ 总投资额 =", np.sum(global_best * A_i))
print("✅ 总选择人数 =", np.sum(global_best))

# === Step 9: Gurobi 优化 ===
print("⚙️ 正在运行 Gurobi 优化...")
model = Model("CVaR_Optimization")
model.setParam("OutputFlag", 0)
model.setParam("TimeLimit", 120)

x = model.addVars(N, vtype=GRB.BINARY, name="x")
eta_var = model.addVar(lb=0, name="eta")
xi_vars = model.addVars(S, lb=0, name="xi")

model.setObjective(
    sum(x[i] * profit_i[i] for i in range(N)) + 10000 * eta_var / R_max, GRB.MAXIMIZE
)
# 信用等级比例约束
for grade, idx_list in group_dict.items():
    if grade in alpha_k:
        model.addConstr(
            sum(x[i] * df.loc[i, 'A_i'] for i in idx_list) <= alpha_k[grade] * B,
            f"Grade_{grade}_Limit"
        )
model.addConstr(sum(x[i] * A_i[i] for i in range(N)) <= B)
model.addConstr(sum(x[i] for i in range(N)) <= m)
model.addConstr(
    sum(x[i] * df.loc[i, 'A_i'] * df.loc[i, 'P_i'] for i in range(N)) <= 15000000,
    "ExpectedRisk"
)

for s in range(S):
    loss_expr = sum(x[i] * A_i[i] * L[i, s] for i in range(N))
    model.addConstr(xi_vars[s] >= loss_expr - eta_var)

model.addConstr(
    eta_var + inv_tail * sum(xi_vars[s] for s in range(S)) <= R_max
)

for i in range(N):
    x[i].Start = int(global_best[i])

model.optimize()

# === Step 10: Gurobi 结果提取 ===
x_array = np.array([x[i].X for i in range(N)])
profit_gurobi = np.sum(profit_i * (x_array > 0.5))

# 风险计算
loss_vector = (L.T @ (x_array * A_i)).flatten()
eta_val = np.percentile(loss_vector, beta * 100)
xi_val = np.maximum(loss_vector - eta_val, 0)
cvar_val = eta_val + inv_tail * np.sum(xi_val)

# 约束验证
actual_total_investment = np.sum(x_array * A_i)
actual_total_selection = np.sum(x_array)
print("\n--- Gurobi 精确求解结果 ---")
print("✅ Gurobi 投资组合期望收益 =", profit_gurobi)
print("✅ Gurobi 风险值 CVaR =", cvar_val)
print("✅ 总投资额 =", actual_total_investment)
print("✅ 总选择人数 =", actual_total_selection)

# === Step 11: 保存完整明细 CSV ===
df['selected_by_gurobi'] = x_array
df['expected_profit'] = df['A_i'] * df['r_i'] * (1 - df['P_i'])

detailed_output = df[['id', 'loan_amnt', 'int_rate', 'estimated_default_prob', 
                      'selected_by_gurobi', 'expected_profit']]
detailed_output.to_csv("code/CVaR_Gurobi_result.csv", index=False)
print("✅ Gurobi详细结果已保存至CVaR_Gurobi_result.csv")

# === Step 12: 保存简要摘要 TXT ===
with open("code/CVaR_summary.txt", "w", encoding="utf-8") as f:
    f.write("✅ Gurobi 最终求解摘要\n")
    f.write("---------------------------------------------------\n")
    f.write(f"总投资额: {actual_total_investment:.2f} / {B:.2f}\n")
    f.write(f"总选择人数: {actual_total_selection} / {m}\n")
    f.write(f"投资组合期望收益: {profit_gurobi:.2f}\n")
    f.write(f"CVaR (β={beta}): {cvar_val:.2f} / {R_max:.2f}\n")
print("✅ Gurobi求解摘要已保存至 CVaR_summary.txt")

