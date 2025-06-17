import pandas as pd
from gurobipy import Model, GRB
import numpy as np

# 1. 读取数据
df = pd.read_excel("code/Input_data.csv")

# 2. 数据预处理
df['A_i'] = df['loan_amnt']
df['r_i'] = df['int_rate'] / 100
df['P_i'] = df['estimated_default_prob']  # 违约概率
df['grade'] = df['grade']

# 3. 清洗数据
df = df[~df[['A_i', 'P_i']].isnull().any(axis=1)]
df = df[~df[['A_i', 'P_i']].apply(lambda col: col.isna() | (col == float('inf')) | (col == float('-inf'))).any(axis=1)]
df = df.reset_index(drop=True)

# 4. 分组与参数
group_dict = df.groupby('grade').groups
N = len(df)

B = 100000000       # 总预算
R_max = 10000000    # 最大风险容忍度（期望损失）
m = 5000            # Top-m 限制
alpha_k = {         # 信用等级投资比例上限
    'A': 0.4, 'B': 0.3, 'C': 0.2,
    'D': 0.1, 'E': 0.05, 'F': 0.02, 'G': 0.01
}

# 5. 构建模型
model = Model("MaxExpectedRepayment")

x = model.addVars(N, vtype=GRB.BINARY, name="x")

# 6. 设置目标函数：最大化 A_i × [r_i × (1 - P_i) ]
model.setObjective(
    sum(x[i] * df.loc[i, 'A_i'] * (df.loc[i, 'r_i'] * (1 - df.loc[i, 'P_i']) ) for i in range(N)),
    GRB.MAXIMIZE
)

# 7. 添加约束

# 预算约束
model.addConstr(sum(x[i] * df.loc[i, 'A_i'] for i in range(N)) <= B, "Budget")

# 信用等级比例约束
for grade, idx_list in group_dict.items():
    if grade in alpha_k:
        model.addConstr(
            sum(x[i] * df.loc[i, 'A_i'] for i in idx_list) <= alpha_k[grade] * B,
            f"Grade_{grade}_Limit"
        )

# 风险约束：期望损失（A_i × P_i）不超过 R_max
model.addConstr(
    sum(x[i] * df.loc[i, 'A_i'] * df.loc[i, 'P_i'] for i in range(N)) <= R_max,
    "ExpectedRisk"
)

# Top-m 限制
model.addConstr(sum(x[i] for i in range(N)) <= m, "TopM")

# 8. 求解模型
model.optimize()

# 9. 输出结果
if model.status == GRB.OPTIMAL:
    selected = [(df.loc[i, 'id'], df.loc[i, 'A_i'], df.loc[i, 'grade'], df.loc[i, 'r_i'], df.loc[i, 'P_i']) 
                for i in range(N) if x[i].X > 0.5]

    result_df = pd.DataFrame(selected, columns=["ID", "Loan_Amount", "Grade", "Interest_Rate", "Default_Prob"])
    result_df.to_csv("code/primal-result.csv", index=False)

    print("✅ 最优贷款组合已保存至 'primal-result.csv'")
    print(f"目标函数值（期望净收益）: {model.objVal:,.2f}")
    print(f"选择的贷款数量: {len(selected)}")
    print(f"总投资金额: {sum(row[1] for row in selected):,.2f}")

    # 按等级统计
    grade_summary = result_df.groupby('Grade').agg({
        'Loan_Amount': ['count', 'sum'],
        'Default_Prob': 'mean'
    }).round(4)
    print("\n按等级统计:")
    print(grade_summary)

else:
    print(f"❌ 模型求解失败，状态码: {model.status}")
    if model.status == GRB.INFEASIBLE:
        print("模型不可行，请检查约束条件")
    elif model.status == GRB.UNBOUNDED:
        print("模型无界，请检查目标函数或约束设置")

# === 保存 Gurobi 求解摘要 ===
with open("code/primal_summary.txt", "w", encoding="utf-8") as f:
    f.write("📌 Gurobi 最优解摘要\n")
    f.write("--------------------------------------------------\n")
    f.write(f"目标函数值（期望净收益）: {model.objVal:,.2f}\n")
    f.write(f"选择的贷款数量: {len(selected)}\n")
    f.write(f"总投资金额: {sum(row[1] for row in selected):,.2f}\n")
    f.write(f"总期望损失: {sum(df.loc[i, 'A_i'] * df.loc[i, 'P_i'] for i in range(N) if x[i].X > 0.5):,.2f}\n")
    f.write("\n📊 按等级统计:\n")
    for grade in sorted(alpha_k.keys()):
        grade_ids = group_dict.get(grade, [])
        selected_ids = [i for i in grade_ids if x[i].X > 0.5]
        total = sum(df.loc[i, 'A_i'] for i in selected_ids)
        count = len(selected_ids)
        ratio = total / B if B > 0 else 0
        f.write(f"  - 等级 {grade}: {count} 个贷款, 投资总额 = {total:,.2f}, 占比 = {ratio:.2%}\n")

print("📄 Gurobi 摘要信息已保存至 'primal_summary.txt'")
