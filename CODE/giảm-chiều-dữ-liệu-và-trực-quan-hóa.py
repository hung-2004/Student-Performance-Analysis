
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =========================
# 1. Đọc dữ liệu
# =========================
df = pd.read_csv(r'Student_Performance_Analysis/DATA/student-mat.csv', sep=',')

# =========================
# 2. Chọn thuộc tính
# =========================
cols = [
    'G1', 'G2', 'G3',
    'studytime', 'failures',
    'absences', 'goout',
    'freetime', 'health'
]

X = df[cols]

# =========================
# 3. Chuẩn hóa dữ liệu
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 4. PCA (giảm về 2 chiều)
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# =========================
# 5. Vẽ PCA
# =========================
plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel('Thành phần chính 1')
plt.ylabel('Thành phần chính 2')
plt.title('PCA: Biểu diễn dữ liệu sau khi giảm chiều')
plt.show()

# =========================
# 6. t-SNE
# =========================
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# =========================
# 7. Vẽ t-SNE
# =========================
plt.figure(figsize=(6,5))
plt.scatter(X_tsne[:,0], X_tsne[:,1])
plt.xlabel('Trục t-SNE 1')
plt.ylabel('Trục t-SNE 2')
plt.title('t-SNE: Trực quan cấu trúc dữ liệu')
plt.show()

print("\n✅ Hoàn thành PCA và t-SNE")