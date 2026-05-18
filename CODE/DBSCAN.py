import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

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
# 4. Giảm chiều để vẽ (PCA)
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# =========================
# 5. Áp dụng DBSCAN
# =========================
dbscan = DBSCAN(eps=0.7, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# =========================
# 6. Thống kê kết quả
# =========================
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise = list(labels).count(-1)

print("Số cụm:", n_clusters)
print("Số điểm nhiễu:", noise)

# =========================
# 7. Vẽ biểu đồ
# =========================
plt.figure(figsize=(6,5))

for label in set(labels):
    if label == -1:
        name = "Nhiễu"
    else:
        name = f"Cụm {label}"
    
    plt.scatter(
        X_pca[labels == label, 0],
        X_pca[labels == label, 1],
        label=name
    )

plt.title("Phân cụm DBSCAN")
plt.xlabel("Thành phần chính 1 (PC1)")
plt.ylabel("Thành phần chính 2 (PC2)")
plt.legend()
plt.show()