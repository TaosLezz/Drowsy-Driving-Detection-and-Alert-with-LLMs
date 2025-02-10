import cv2
from scipy.spatial import distance
import mediapipe as mp

# Khởi tạo face mesh từ mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Hàm tính tỉ số khía mắt (Eye Aspect Ratio - EAR)
def eye_aspect_ratio(eye):
    # eye: danh sách 6 điểm theo thứ tự:
    # [0]: góc ngoài, [1]: mép trên bên trong, [2]: mép trên bên ngoài,
    # [3]: góc trong, [4]: mép dưới bên ngoài, [5]: mép dưới bên trong
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Hàm tính tỉ số khía miệng (Mouth Aspect Ratio - MAR)
def mouth_aspect_ratio(face_landmarks):
    # Lấy các điểm từ landmark theo chỉ số được định nghĩa
    #  - Góc trái: 61, góc phải: 291
    #  - Trung tâm trên môi: 0, trung tâm dưới môi: 17
    left_corner = [face_landmarks.landmark[61].x, face_landmarks.landmark[61].y]
    right_corner = [face_landmarks.landmark[291].x, face_landmarks.landmark[291].y]
    top_lip = [face_landmarks.landmark[0].x, face_landmarks.landmark[0].y]
    bottom_lip = [face_landmarks.landmark[17].x, face_landmarks.landmark[17].y]

    # Tính khoảng cách theo phương dọc và ngang
    vertical_dist = distance.euclidean(top_lip, bottom_lip)
    horizontal_dist = distance.euclidean(left_corner, right_corner)
    mar = vertical_dist / horizontal_dist
    return mar

# Hàm xử lý ảnh và xác định trạng thái “buồn ngủ”
def process_image(frame):
    # Định nghĩa ngưỡng cho mắt và miệng
    EYE_AR_THRESH = 0.25    # Nếu EAR nhỏ hơn ngưỡng -> mắt khép
    MOUTH_AR_THRESH = 0.6   # Nếu MAR lớn hơn ngưỡng -> miệng há (có thể là dấu hiệu ngáp)

    if frame is None:
        raise ValueError('Image is not found or unable to open')

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Cờ báo hiệu trạng thái buồn ngủ
    drowsy = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Lấy 6 điểm cho mắt phải theo thứ tự:
            # [góc ngoài, mép trên bên trong, mép trên bên ngoài, góc trong, mép dưới bên ngoài, mép dưới bên trong]
            right_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye = []
            for i in right_eye_indices:
                right_eye.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])

            # Lấy 6 điểm cho mắt trái
            left_eye_indices = [362, 385, 387, 263, 373, 380]
            left_eye = []
            for i in left_eye_indices:
                left_eye.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])

            # Tính EAR cho từng mắt và trung bình lại
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Tính MAR cho miệng
            mar = mouth_aspect_ratio(face_landmarks)

            # In ra để kiểm tra giá trị (tuỳ chọn)
            # print(f"EAR: {ear:.2f}, MAR: {mar:.2f}")

            # Nếu mắt khép (EAR thấp) hoặc miệng mở (MAR cao) thì đánh dấu buồn ngủ.
            if ear < EYE_AR_THRESH or mar > MOUTH_AR_THRESH:
                drowsy = True
            else:
                drowsy = False

    return drowsy

# Ví dụ chạy code với webcam (hoặc một frame ảnh)
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Kiểm tra trạng thái buồn ngủ
        is_drowsy = process_image(frame)

        # Hiển thị kết quả lên frame
        status_text = "DROWSY" if is_drowsy else "ALERT"
        cv2.putText(frame, status_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_drowsy else (0, 255, 0), 2)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
