import pandas as pd
import numpy as np
import glob
import sys

print("--- Bắt đầu tạo sinh dữ liệu (Augmentation) ---")

# --- CÀI ĐẶT ---
AUGMENTATION_FACTOR = 15 # Sẽ tạo 15 bản sao cho MỖI mẫu gốc
OUTPUT_FILE = 'data_augmented.csv' # Tên file dữ liệu khổng lồ
# ----------------

# --- ĐỘ MẠNH CỦA NHIỄU (Có thể điều chỉnh) ---
JITTER_STRENGTH = 0.003    # Giả lập rung tay (rất nhỏ)
TRANSLATE_STRENGTH = 0.02  # Giả lập di chuyển (vừa phải)
SCALE_STRENGTH = 0.05      # Giả lập phóng to/thu nhỏ
# ----------------

# 1. Đọc và gộp tất cả các file .csv gốc
# (Tìm file bắt đầu bằng 'hand_data' hoặc tên gốc của bạn, ví dụ 'data_loop_1')
data_files = glob.glob('hand_data*.csv') + glob.glob('data_loop*.csv')

if not data_files:
    print("Lỗi: Không tìm thấy file .csv gốc (vd: 'hand_data.csv').")
    sys.exit()

print(f"Tìm thấy các file dữ liệu gốc: {data_files}")
list_of_dataframes = [pd.read_csv(f) for f in data_files if not pd.read_csv(f).empty]
if not list_of_dataframes:
    print("Lỗi: Không đọc được dữ liệu nào.")
    sys.exit()

original_data = pd.concat(list_of_dataframes, ignore_index=True)
print(f"Tổng số mẫu gốc: {len(original_data)}")

# Tách X (42 cột) và y (nhãn)
try:
    X_raw = original_data.drop('label', axis=1)
    y_raw = original_data['label']
    column_names = list(X_raw.columns) + ['label']
    num_coords = 21 # 21 tọa độ x, 21 tọa độ y
except KeyError:
    print("Lỗi: Không tìm thấy cột 'label' trong file .csv gốc.")
    sys.exit()

# 2. Bắt đầu tạo dữ liệu nhiễu
augmented_data_list = []
total_samples = len(X_raw)

print(f"Đang tạo {total_samples * AUGMENTATION_FACTOR} mẫu nhiễu mới...")

for i in range(total_samples):
    original_sample = X_raw.iloc[i].values 
    label = y_raw.iloc[i]
    
    if (i + 1) % 1000 == 0:
        print(f"Đã xử lý {i+1}/{total_samples} mẫu gốc...")

    # Tạo 15 bản sao nhiễu từ mẫu gốc này
    for _ in range(AUGMENTATION_FACTOR):
        new_sample = original_sample.copy()
        
        # a) Thêm Jitter (Rung tay)
        jitter = np.random.uniform(-JITTER_STRENGTH, JITTER_STRENGTH, size=new_sample.shape)
        new_sample += jitter
        
        # b) Thêm Translation (Dịch chuyển)
        trans_x = np.random.uniform(-TRANSLATE_STRENGTH, TRANSLATE_STRENGTH)
        trans_y = np.random.uniform(-TRANSLATE_STRENGTH, TRANSLATE_STRENGTH)
        new_sample[:num_coords] += trans_x # Cộng vào 21 cột X
        new_sample[num_coords:] += trans_y # Cộng vào 21 cột Y
        
        # c) Thêm Scaling (Thu phóng)
        scale = np.random.uniform(1 - SCALE_STRENGTH, 1 + SCALE_STRENGTH)
        new_sample *= scale
        
        new_row = np.append(new_sample, label)
        augmented_data_list.append(new_row)

print("Đã tạo xong dữ liệu nhiễu.")

# 3. Gộp dữ liệu gốc và dữ liệu nhiễu
print("Đang gộp dữ liệu...")
augmented_df = pd.DataFrame(augmented_data_list, columns=column_names)
final_data = pd.concat([original_data, augmented_df], ignore_index=True)

# 4. Lưu file
final_data.to_csv(OUTPUT_FILE, index=False)
print(f"--- HOÀN TẤT! ---")
print(f"File dữ liệu gốc có: {len(original_data)} mẫu")
print(f"File dữ liệu nhiễu có: {len(augmented_df)} mẫu")
print(f"Đã lưu file tổng hợp '{OUTPUT_FILE}' với {len(final_data)} mẫu.")