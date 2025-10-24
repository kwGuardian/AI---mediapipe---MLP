import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
import pickle
import sys
import matplotlib.pyplot as plt

print("--- Bắt đầu Giai đoạn 2: Huấn luyện (train_3.py - trên Data Augmented) ---")

# --- 1. ĐỌC DỮ LIỆU ĐÃ TẠO SINH ---
DATA_FILE = 'data_augmented.csv' # <<< ĐỌC TỪ FILE MỚI

try:
    data = pd.read_csv(DATA_FILE)
    if data.empty:
        print(f"Lỗi: File '{DATA_FILE}' bị trống.")
        sys.exit()
    print(f"Đã đọc file '{DATA_FILE}' thành công. Tổng số mẫu: {len(data)}")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{DATA_FILE}'.")
    print("Bạn phải chạy file 'augment_data.py' trước để tạo ra file này.")
    sys.exit()
except Exception as e:
    print(f"Lỗi khi đọc file: {e}")
    sys.exit()

# --- 2. TÁCH VÀ TIỀN XỬ LÝ DỮ LIỆU ---
# (Phần còn lại của file train_3.py giữ nguyên)
try:
    X_raw = data.drop('label', axis=1) 
    y_raw = data['label']
except KeyError:
    print("Lỗi 'KeyError': Không tìm thấy cột 'label'.")
    sys.exit()

# a) Chuẩn hóa (X)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw) 

# b) Mã hóa nhãn (y)
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_) 

print(f"Đã chuẩn hóa X và mã hóa y (Số lớp: {num_classes}).")
    
# 3. Chia 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Đã chia dữ liệu Train/Test...")

# 4. Khởi tạo mô hình (MLP)
model = Sequential([
    InputLayer(shape=(X_train.shape[1],)), # 42
    Dense(128, activation='relu'),         
    Dropout(0.4),                          
    Dense(64, activation='relu'),          
    Dropout(0.4),
    Dense(num_classes, activation='softmax') # 6
])

# 5. Biên dịch mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.summary() 

# 6. Huấn luyện
print("Đang huấn luyện (Neural Network)... (Sẽ mất nhiều thời gian hơn)")
history = model.fit(X_train, y_train, 
                    epochs=40,
                    batch_size=32, 
                    validation_data=(X_test, y_test))
print("Huấn luyện xong!")

# 7. Vẽ biểu đồ
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy (Augmented)')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss (Augmented)')
plt.savefig('training_history_3_AUGMENTED.png') # Lưu file ảnh mới
print("Đã lưu biểu đồ GỐC vào 'training_history_3_AUGMENTED.png'")

# 8. Lưu kết quả
model.save('gesture_model_3.keras') # Ghi đè "não" cũ
print("Đã lưu 'não' GỐC vào 'gesture_model_3.keras'")
with open('processors_3.pkl', 'wb') as f:
    pickle.dump((scaler, le), f) # Ghi đè "bộ xử lý" cũ
print("Đã lưu 'bộ xử lý' (scaler, le) vào 'processors_3.pkl'")
print("--- Kết thúc Giai đoạn 2 (Augmented) ---")