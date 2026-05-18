import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Đọc dữ liệu
# =========================
df = pd.read_csv(r'Student_Performance_Analysis/DATA/student-mat.csv')

# =========================
# 2. Kiểm tra dữ liệu
# =========================
print("5 dòng đầu của dữ liệu:")
print(df.head())

print("\nThông tin dữ liệu:")
print(df.info())

# =========================
# 3. Kiểm tra Missing Values
# =========================
print("\nSố lượng giá trị thiếu:")
print(df.isnull().sum())

# =========================
# 4. Chọn thuộc tính dùng để phân cụm
# =========================
features = [
    'G1', 'G2', 'G3',
    'studytime', 'failures',
    'absences', 'goout',
    'freetime', 'health'
]

X = df[features]

# =========================
# 5. Chuẩn hóa dữ liệu
# =========================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Chuyển về DataFrame để dễ xem
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# =========================
# 6. Hiển thị kết quả
# =========================
print("\nDữ liệu sau khi chuẩn hóa:")
print(X_scaled_df.head())