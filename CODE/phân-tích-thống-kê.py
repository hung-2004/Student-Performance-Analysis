
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Đọc dữ liệu
# =========================
df = pd.read_csv(r'Student_Performance_Analysis/DATA/student-mat.csv', sep=',')

# =========================
# 2. Chọn các thuộc tính số
# =========================
cols = [
    'G1', 'G2', 'G3',
    'studytime', 'failures',
    'absences', 'goout',
    'freetime', 'health'
]

data = df[cols]

# =========================
# 3. HISTOGRAM
# =========================
plt.figure(figsize=(15,10))

for i, col in enumerate(cols):
    plt.subplot(3, 3, i+1)
    plt.hist(data[col], bins=10)
    plt.title(f'Histogram - Phân bố của {col}')
    plt.xlabel(col)
    plt.ylabel('Tần suất')

plt.suptitle('Histogram: Phân bố dữ liệu các thuộc tính', fontsize=16)
plt.tight_layout()
plt.show()

# =========================
# 4. HEATMAP
# =========================
corr = data.corr()

plt.figure(figsize=(10,8))
plt.imshow(corr)
plt.colorbar()

plt.xticks(range(len(cols)), cols, rotation=45)
plt.yticks(range(len(cols)), cols)

plt.title('Heatmap: Ma trận tương quan giữa các thuộc tính')

for i in range(len(cols)):
    for j in range(len(cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha='center', va='center')

plt.tight_layout()
plt.show()

# =========================
# 5. SCATTER PLOT
# =========================

# G1 vs G2
plt.figure()
plt.scatter(df['G1'], df['G2'])
plt.xlabel('Điểm kỳ 1 (G1)')
plt.ylabel('Điểm kỳ 2 (G2)')
plt.title('Scatter Plot: Mối quan hệ giữa G1 và G2')
plt.show()

# G2 vs G3
plt.figure()
plt.scatter(df['G2'], df['G3'])
plt.xlabel('Điểm kỳ 2 (G2)')
plt.ylabel('Điểm cuối (G3)')
plt.title('Scatter Plot: Mối quan hệ giữa G2 và G3')
plt.show()

# Absences vs G3
plt.figure()
plt.scatter(df['absences'], df['G3'])
plt.xlabel('Số buổi nghỉ')
plt.ylabel('Điểm cuối (G3)')
plt.title('Scatter Plot: Ảnh hưởng của nghỉ học đến kết quả')
plt.show()

# Studytime vs G3
plt.figure()
plt.scatter(df['studytime'], df['G3'])
plt.xlabel('Thời gian học')
plt.ylabel('Điểm cuối (G3)')
plt.title('Scatter Plot: Thời gian học và kết quả')
plt.show()

print("\n✅ Đã hiển thị đầy đủ: Histogram, Heatmap, Scatter Plot")