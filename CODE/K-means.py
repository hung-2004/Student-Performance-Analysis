import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
# 3. Chuẩn hóa
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 4. PCA để vẽ
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# =========================
# 5. HÀM ĐẶT TÊN CỤM
# =========================
def gan_ten_cum(df, labels):
    df_temp = df.copy()
    df_temp['Cluster'] = labels

    # Tính trung bình điểm G3 của từng cụm
    mean_g3 = df_temp.groupby('Cluster')['G3'].mean().sort_values()

    # Tên cụm theo thứ tự điểm tăng dần
    names = ['Nguy cơ', 'Trung bình', 'Học tốt']

    cluster_names = {}

    for i, cluster in enumerate(mean_g3.index):
        cluster_names[cluster] = names[i]

    return cluster_names

# =========================
# 6. HÀM VẼ BIỂU ĐỒ
# =========================
def ve_kmeans(k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # đặt tên cụm
    cluster_names = gan_ten_cum(df, labels)
    
    plt.figure(figsize=(6,5))
    
    for cluster in set(labels):
        plt.scatter(
            X_pca[labels == cluster, 0],
            X_pca[labels == cluster, 1],
            label=cluster_names[cluster]
        )
    
    plt.title(f'K-Means (k={k})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()

# =========================
# 7. CHẠY
# =========================
ve_kmeans(3)
