import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# =========================
# 1. Đọc dữ liệu
# =========================
df = pd.read_csv(r'Student_Performance_Analysis/DATA/student-mat.csv')

# =========================
# 2. Chọn thuộc tính
# =========================
features = [
    'G1','G2','G3',
    'studytime','failures',
    'absences','goout',
    'freetime','health'
]

X = df[features]

# =========================
# 3. Chuẩn hóa dữ liệu
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 4. Tính linkage (Ward)
# =========================
Z = linkage(X_scaled, method='ward')

# =========================
# 5. Vẽ Dendrogram
# =========================
plt.figure(figsize=(12,6))

dendrogram(
    Z,
    truncate_mode='level',  # rút gọn cây
    p=5                     # hiển thị 5 cấp
)

plt.title('Dendrogram - Phân cụm phân cấp')
plt.xlabel('Sinh viên')
plt.ylabel('Khoảng cách')

plt.show()