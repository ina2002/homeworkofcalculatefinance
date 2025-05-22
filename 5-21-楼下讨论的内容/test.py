import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 读取数据
df = pd.read_excel("工作簿1.xlsx")

# 2. sub_grade 有序映射
sub_order = ['A1', 'A2', 'A3', 'A4', 'A5',
             'B1', 'B2', 'B3', 'B4', 'B5',
             'C1', 'C2', 'C3', 'C4', 'C5',
             'D1', 'D2', 'D3', 'D4', 'D5',
             'E1', 'E2', 'E3', 'E4', 'E5']
sub_rank = {k: i+1 for i, k in enumerate(sub_order)}
df['sub_grade_score'] = df['sub_grade'].map(sub_rank)

# ✅ 加这一行：删除 NaN 标签行（可能有无法映射的 sub_grade）
df = df.dropna(subset=['sub_grade_score'])

# 3. 特征列
features = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
            'annual_inc', 'purpose', 'dti', 'delinq_2yrs', 'inq_last_6mths',
            'last_fico_range_high', 'last_fico_range_low']
X = df[features].copy()
y = df['sub_grade_score']

# 4. 类别变量编码
for col in ['term', 'grade', 'sub_grade', 'purpose']:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# 5. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. 拆分数据
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. 模型训练
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. 预测 + 得分反向归一化为“违约概率”
y_pred = model.predict(X_scaled)
y_pred_prob = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
df['predicted_subgrade_score'] = y_pred
df['estimated_default_prob'] = 1 - y_pred_prob  # 值越高 → 越接近E5 → 越可能违约

# 9. 可视化趋势
avg_probs = df.groupby('sub_grade')['estimated_default_prob'].mean().sort_index()
plt.figure(figsize=(10, 5))
avg_probs.plot(kind='bar')
plt.title("平均违约概率（监督模型） vs sub_grade")
plt.ylabel("estimated_default_prob")
plt.xlabel("sub_grade")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 10. 保存结果
df.to_excel("loans_supervised_default_prob.xlsx", index=False)
print("✅ 文件已保存为 loans_supervised_default_prob.xlsx")
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
spearman_corr, _ = spearmanr(y, y_pred)

print(f"📊 MSE: {mse:.4f}")
print(f"📈 R^2 Score: {r2:.4f}")
print(f"📉 Spearman相关系数（排序一致性）: {spearman_corr:.4f}")
