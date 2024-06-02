import cv2
import mediapipe as mp

def main():
    # Mediapipe Face va Hands modellarni yuklash
    mp_face_detection = mp.solutions.face_detection
    mp_hands = mp.solutions.hands
    face_detection = mp_face_detection.FaceDetection()
    hands = mp_hands.Hands()

    # Videofaylni yoki kamerani boshlash
    cap = cv2.VideoCapture(0)  # 0 - tovushli kamera, 1 - USB-kamera

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Kameraga kirish muammo ro'y berdi. Chiqish...")
            break

        # Ranglar modelini BGR dan RGB ga o'tkazish
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Yuzlarni aniqlash
        face_results = face_detection.process(frame_rgb)
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)

        # Qo'l harakatlarini aniqlash
        hands_results = hands.process(frame_rgb)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Qo'l landmarklarini chizish
                for lm in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Natijani chiqarish
        cv2.imshow('Yuz va Qo\'l Harakatlari Kuzatish', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
