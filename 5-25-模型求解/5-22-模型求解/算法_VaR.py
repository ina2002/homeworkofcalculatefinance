import pandas as pd
import numpy as np

# === Step 1: 数据读取与处理 ===
df = pd.read_excel("Input_data.xlsx", sheet_name="Sheet1")

df['A_i'] = df['loan_amnt']
df['P_i'] = 1 - df['estimated_default_prob']
df['r_i'] = df['int_rate'] / 100
df['profit'] = df['A_i'] * (df['r_i'] * (1 - df['P_i']) - df['P_i'])

df = df[(df['P_i'] >= 0) & (df['P_i'] <= 1) & (~df['P_i'].isna())].copy()

A_i = df['A_i'].values
P_i = df['P_i'].values
profit_i = df['profit'].values
N = len(df)

# === Step 2: 蒙特卡洛模拟 l_i(s) ~ Bernoulli(P_i) ===
S = 1000
np.random.seed(42)
L = np.random.binomial(n=1, p=P_i[:, None], size=(N, S))

# === Step 3: 参数设置 ===
B = 100000000
m = 5000
beta = 0.95
VaR_alpha = int((1 - beta) * S)

# === Step 4: 粒子群优化参数 ===
pop_size = 30
max_iter = 100
w = 0.7
c1 = 1.5
c2 = 1.5

# === Step 5: 启发式初始化粒子 ===
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

# === Step 6: 适应度函数定义 ===
def evaluate(x):
    x = x.astype(int)
    if np.sum(x) > m:
        return -1e10
    total_budget = np.sum(x * A_i)
    if total_budget > B:
        return -1e10
    losses = np.dot((x * A_i), L)
    sorted_losses = np.sort(losses)
    eta = sorted_losses[VaR_alpha]
    if np.sum(losses > eta) > VaR_alpha:
        return -1e10
    return np.sum(x * profit_i)

# === Step 7: 粒子群主循环 ===
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

# === Step 8: 输出最终结果 ===
selected_indices = np.where(global_best == 1)[0]
selected_ids = df.iloc[selected_indices]['id'].tolist()
selected_profits = profit_i[selected_indices].sum()

print("✅ 选中的借款人 ID：", selected_ids)
print("✅ 最优组合期望收益 =", selected_profits)
