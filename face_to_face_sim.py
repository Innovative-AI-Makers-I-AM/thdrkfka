import dlib
import cv2
import numpy as np

# Dlib의 얼굴 탐지기와 랜드마크 예측기, 얼굴 인식 모델을 로드합니다.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# 얼굴 유사도 계산 함수
def calculate_similarity(face1_descriptor, face2_descriptor):
    # 두 얼굴 간의 유사도를 계산합니다.
    distance = np.linalg.norm(np.array(face1_descriptor) - np.array(face2_descriptor))
    return distance

# 웹캠 비디오 캡처를 시작합니다.
cap = cv2.VideoCapture(0)

# 출입을 허용할 유사도 임계값을 설정합니다.
threshold = 0.6

# 등록된 얼굴 이미지를 로드합니다.
registered_image = cv2.imread("사람.jpg")
registered_gray = cv2.cvtColor(registered_image, cv2.COLOR_BGR2GRAY)
registered_faces = detector(registered_gray)

# 등록된 얼굴이 존재하지 않으면 프로그램을 종료합니다.
if len(registered_faces) != 1:
    print("등록된 얼굴을 찾을 수 없거나 여러 얼굴이 있습니다.")
    exit()

# 등록된 얼굴의 랜드마크를 예측합니다.
registered_landmarks = predictor(registered_gray, registered_faces[0])

# 등록된 얼굴의 특징 벡터를 계산합니다.
registered_descriptor = facerec.compute_face_descriptor(registered_image, registered_landmarks)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴을 탐지합니다.
    faces = detector(gray)

    for face in faces:
        # 얼굴 영역 추출
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        face_image = frame[y1:y2, x1:x2]

        # 얼굴 영역에 사각형을 그립니다.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 얼굴의 랜드마크를 예측합니다.
        landmarks = predictor(gray, face)

        # 얼굴을 등록된 얼굴과 비교합니다.
        face_descriptor = facerec.compute_face_descriptor(face_image, landmarks)
        similarity = calculate_similarity(registered_descriptor, face_descriptor)

        # 출입 허용 여부를 판단합니다.
        if similarity <= threshold:
            cv2.putText(frame, "Access Granted", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Access Denied", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 프레임을 출력합니다.
    cv2.imshow("Frame", frame)

    # 'q' 키를 누르면 루프를 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처와 모든 창을 종료합니다.
cap.release()
cv2.destroyAllWindows()
