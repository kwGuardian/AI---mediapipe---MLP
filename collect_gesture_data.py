import cv2
import mediapipe as mp
import csv
import time

# --- Thiết lập ---
SAMPLES_PER_LABEL = 1000  # Đã tăng từ 200 lên 1000
# ------------------

print("Mo webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Khong mo duoc webcam")
    exit()  # Thoát nếu không mở được
else:
    print("Webcam mo thanh cong")
    
# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Danh sách nhãn (6 hành động)
labels = ["bat_loa", "tat_loa", "tang_am", "giam_am", "prev_song", "next_song"]

# Tạo file CSV lưu dữ liệu
# Sử dụng 'with open' để tự động quản lý file
try:
    with open('hand_data_1.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Viết tiêu đề cột
        writer.writerow([f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + ['label'])

        for label in labels:
            print(f"\nChuẩn bị thu dữ liệu cho cử chỉ: {label}")
            print("Dat tay vao vi tri, bat dau sau 5 giay...")
            cv2.waitKey(5000) # Tạm dừng 5 giây (thay vì time.sleep để cv2 không bị đơ)
            
            print(f"Bat dau thu du lieu cho '{label}'...")

            count = 0
            while count < SAMPLES_PER_LABEL: # Thu thập đủ 1000 mẫu
                ret, frame = cap.read()
                if not ret:
                    print("Loi khung hinh, bo qua")
                    continue
                
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                
                if result.multi_hand_landmarks:
                    for handLms in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                        
                        data = []
                        
                        # Lấy tọa độ gốc (cổ tay - điểm 0)
                        x0 = handLms.landmark[0].x
                        y0 = handLms.landmark[0].y
                        
                        # Tính toán và thêm tọa độ tương đối
                        for lm in handLms.landmark:
                            data.append(lm.x - x0)
                        for lm in handLms.landmark:
                            data.append(lm.y - y0)
                            
                        data.append(label)
                        
                        # Ghi vào file
                        writer.writerow(data)
                        count += 1
                
                # Hiển thị thông tin
                cv2.putText(frame, f'Dang thu: {label} - Mau: {count}/{SAMPLES_PER_LABEL}', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Collecting Data", frame)

                # --- ĐÃ THÊM ĐỘ TRỄ 50MS TẠI ĐÂY ---
                # Nhấn Q để thoát sớm, đồng thời tạo độ trễ 50ms
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    print("Da dung thu du lieu som!")
                    break # Thoát vòng lặp while
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # Giữ nguyên waitKey(1) ở đây để vòng for lặp nhanh
                break # Thoát vòng lặp for
            
            print(f"Da thu xong {count} khung hinh cho '{label}'")

except IOError as e:
    print(f"Loi khi mo/ghi file: {e}")
finally:
    # Luôn giải phóng tài nguyên sau khi xong
    cap.release()
    cv2.destroyAllWindows()
    print("\nHoan tat! Du lieu duoc luu vao 'hand_data_1.csv'")