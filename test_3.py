import cv2
import mediapipe as mp
import pickle
import numpy as np
import time 
import tensorflow as tf

print("--- Bắt đầu Giai đoạn 3: Kiểm tra (test_3.py - Neural Network) ---")

# Tải mô hình và bộ xử lý
try:
    # a) Tải mô hình Keras
    model = tf.keras.models.load_model('gesture_model_3.keras')
    # b) Tải scaler và label encoder
    with open('processors_3.pkl', 'rb') as f:
        scaler, le = pickle.load(f)
    print("Đã tải mô hình 'gesture_model_3.keras' và 'processors_3.pkl' thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình hoặc bộ xử lý: {e}")
    print("Bạn phải chạy file 'train_3.py' trước để tạo ra các file này.")
    exit()

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Lỗi: Không mở được webcam")
    exit()

print("Bắt đầu nhận diện... (Nhấn Q để thoát)")

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không đọc được khung hình từ webcam.")
        break
        
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    frame.flags.writeable = False
    result = hands.process(rgb)
    frame.flags.writeable = True
    
    prediction_text = "" 

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            
            data = []
            
            # --- XỬ LÝ DỮ LIỆU NHẤT QUÁN (TƯƠNG ĐỐI) ---
            x0 = handLms.landmark[0].x
            y0 = handLms.landmark[0].y
            for lm in handLms.landmark: data.append(lm.x - x0)
            for lm in handLms.landmark: data.append(lm.y - y0)
            # --------------------------------
                
            try:
                data_array = np.array(data).reshape(1, -1) 
                
                # --- DỰ ĐOÁN BẰNG NEURAL NETWORK ---
                # 1. Dùng scaler đã lưu để chuẩn hóa dữ liệu mới
                data_scaled = scaler.transform(data_array)
                
                # 2. Dự đoán (sẽ ra 6 xác suất)
                probabilities = model.predict(data_scaled, verbose=0)
                
                # 3. Lấy chỉ số (index) có xác suất cao nhất
                prediction_index = np.argmax(probabilities)
                
                # 4. Dùng 'le' đã lưu để dịch (index) -> (chữ)
                prediction_text = le.inverse_transform([prediction_index])[0]
                # -----------------------------
            
            except Exception as e:
                print(f"Lỗi khi dự đoán: {e}")
                prediction_text = "ERROR"

    # Hiển thị FPS
    end_time = time.time()
    total_time = end_time - start_time
    if total_time > 0:
        fps = 1.0 / total_time
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Hiển thị dự đoán
    cv2.putText(frame, prediction_text, (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Test Model 3 (MLP) - (Press Q to exit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp
cap.release()
cv2.destroyAllWindows()
print("Chương trình kiểm tra đã kết thúc.")
