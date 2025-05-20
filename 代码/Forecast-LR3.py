import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def bootstrap_ci(p_array, n_samples=1000, n_obs=100):
    lower_bounds, upper_bounds = [], []
    for p in p_array:
        p = np.clip(p, 0, 1)  # 确保p在0到1之间
        samples = np.random.binomial(n=n_obs, p=p, size=n_samples) / n_obs
        lower, upper = np.percentile(samples, [2.5, 97.5])
        lower_bounds.append(lower)
        upper_bounds.append(upper)
    return np.array(lower_bounds), np.array(upper_bounds)


# 读取数据并映射
df = pd.read_excel("Data1.xlsx")
mapping = {'A1': 0.10, 'A2': 0.13, 'A3': 0.16, 'A4': 0.19, 'A5': 0.22,
           'B1': 0.25, 'B2': 0.28, 'B3': 0.31, 'B4': 0.34, 'B5': 0.37,
           'C1': 0.40, 'C2': 0.43, 'C3': 0.46, 'C4': 0.49, 'C5': 0.52,
           'D1': 0.55, 'D2': 0.58, 'D3': 0.61, 'D4': 0.64, 'D5': 0.67,
           'E1': 0.70, 'E2': 0.73, 'E3': 0.76, 'E4': 0.79, 'E5': 0.82,
           'F1': 0.85, 'F2': 0.87, 'F3': 0.89, 'F4': 0.91, 'F5': 0.93,
           'G1': 0.95, 'G2': 0.96, 'G3': 0.97, 'G4': 0.98, 'G5': 0.99}
df['pi_mapped'] = df['sub_grade'].map(mapping)
df['term_num'] = df['term'].str.extract(r'(\d+)').astype(int)

features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
            'delinq_2yrs', 'inq_last_6mths', 'total_acc', 'fico', 'term_num']
X = df[features].fillna(0)
y = df['pi_mapped']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X)

lower_bounds, upper_bounds = bootstrap_ci(y_pred)

df['predicted_pi'] = y_pred
df['ci_lower'] = lower_bounds
df['ci_upper'] = upper_bounds

df[['id', 'sub_grade', 'pi_mapped', 'predicted_pi', 'ci_lower', 'ci_upper']].to_excel("LinearRegression_Pi_CI.xlsx", index=False)
print("预测结果已保存")
