import pandas as pd
import numpy as np
from gurobipy import Model, GRB

# 读取数据
df = pd.read_excel("Input_data.xlsx", sheet_name="Sheet1")

# 构造必要参数
df['A_i'] = df['loan_amnt']
df['P_i'] = 1 - df['estimated_default_prob']  # 转换为违约概率
df['r_i'] = df['int_rate'] / 100  # 转为小数

N = len(df)
B = 1e8  # 预算 1 亿
beta = 0.95  # 置信水平
S = 1000  # 模拟场景数
R_max = 1.5e7  # CVaR 上限
# 清洗违约概率：确保在 [0, 1] 范围内且非空
# 清洗并修正 N
df = df[df['estimated_default_prob'].notna()].copy()
df['P_i'] = (1 - df['estimated_default_prob']).clip(0, 1)
df['A_i'] = df['loan_amnt']
df['r_i'] = df['int_rate'] / 100

N = len(df)  # 🔁 更新 N！


# 蒙特卡洛模拟损失
np.random.seed(42)
loss_matrix = np.random.binomial(1, df['P_i'].values[:, None], size=(N, S)) * df['A_i'].values[:, None]

# 创建模型
model = Model("CVaR_Model")
x = model.addVars(N, vtype=GRB.BINARY, name="x")
eta = model.addVar(vtype=GRB.CONTINUOUS, name="eta")
xi = model.addVars(S, vtype=GRB.CONTINUOUS, lb=0.0, name="xi")

# 目标函数：最大化期望收益
model.setObjective(
    sum(x[i] * df.loc[i, 'A_i'] * (df.loc[i, 'r_i'] * (1 - df.loc[i, 'P_i']) - df.loc[i, 'P_i']) for i in range(N)),
    GRB.MAXIMIZE
)

# 添加约束
model.addConstr(sum(x[i] * df.loc[i, 'A_i'] for i in range(N)) <= B, name="budget")

for s in range(S):
    loss_s = sum(x[i] * loss_matrix[i, s] for i in range(N))
    model.addConstr(xi[s] >= loss_s - eta, name=f"cvar_excess_{s}")

model.addConstr(
    eta + (1 / ((1 - beta) * S)) * sum(xi[s] for s in range(S)) <= R_max,
    name="cvar_limit"
)

# 求解模型
model.optimize()

# 输出选择结果
selected = [(int(df.loc[i, 'id']), df.loc[i, 'A_i']) for i in range(N) if x[i].X > 0.5]
print(f"选中的借款人数量：{len(selected)}")
print(f"投资总额：{sum([a for _, a in selected]):,.2f}")
print("前10个借款人ID：", [i for i, _ in selected[:10]])
