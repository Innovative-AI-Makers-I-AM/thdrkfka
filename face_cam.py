import dlib
import cv2
import numpy as np

# Dlib의 얼굴 탐지기와 랜드마크 예측기를 로드합니다.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 비디오 캡처를 시작합니다.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴을 탐지합니다.
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # 얼굴 영역에 사각형을 그립니다.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 랜드마크를 예측합니다.
        landmarks = predictor(gray, face)

        # 랜드마크를 프레임에 그립니다.
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # 프레임을 출력합니다.
    cv2.imshow("Frame", frame)

    # 'q' 키를 누르면 루프를 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처와 모든 창을 종료합니다.
cap.release()
cv2.destroyAllWindows()