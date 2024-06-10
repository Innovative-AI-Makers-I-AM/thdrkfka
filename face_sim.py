import dlib
import cv2
import numpy as np

# dlib의 얼굴 탐지기, 랜드마크 예측기, 얼굴 인식 모델을 로드합니다.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# 이미지 불러오기
image1 = cv2.imread("karina.jpg")
image2 = cv2.imread("suzy.jpg")

# image2 = cv2.imread("karina2.jpg")

# 그레이스케일로 변환
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 이미지에서 얼굴 탐지
faces1 = detector(gray1)
faces2 = detector(gray2)

# 얼굴이 각 이미지에서 하나씩만 존재하는지 확인
if len(faces1) != 1 or len(faces2) != 1:
    print("얼굴을 찾을 수 없거나 여러 얼굴이 있습니다.")
    exit()

# 랜드마크 예측
face1_landmarks = predictor(gray1, faces1[0])
face2_landmarks = predictor(gray2, faces2[0])

# 얼굴 인식
face_descriptor1 = facerec.compute_face_descriptor(image1, face1_landmarks)
face_descriptor2 = facerec.compute_face_descriptor(image2, face2_landmarks)

# 두 얼굴 간의 유사도 계산
distance = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))

print("두 얼굴의 유사도:", distance)


# import dlib
# import cv2
# import numpy as np

# # Dlib의 얼굴 탐지기와 랜드마크 예측기를 로드합니다.
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("dlib_face_recognition_resnet_model_v1.dat")

# # 얼굴 유사도 계산 함수
# def calculate_similarity(face1, face2):
#     # 두 얼굴의 랜드마크 포인트 간의 거리 계산
#     landmarks1 = np.array([(landmark.x, landmark.y) for landmark in face1.parts()])
#     landmarks2 = np.array([(landmark.x, landmark.y) for landmark in face2.parts()])
#     distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
#     # 평균 거리를 반환하여 유사도 계산
#     return np.mean(distances)

# # 이미지 불러오기
# image1 = cv2.imread("karina.jpg")
# image2 = cv2.imread("karina2.jpg")

# # 그레이스케일로 변환
# gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# # 이미지에서 얼굴 탐지
# faces1 = detector(gray1)
# faces2 = detector(gray2)

# # 얼굴이 각 이미지에서 하나씩만 존재하는지 확인
# if len(faces1) != 1 or len(faces2) != 1:
#     print("얼굴을 찾을 수 없거나 여러 얼굴이 있습니다.")
#     exit()

# # 랜드마크 예측
# face1_landmarks = predictor(gray1, faces1[0])
# face2_landmarks = predictor(gray2, faces2[0])

# # 두 얼굴의 유사도 계산
# similarity = calculate_similarity(face1_landmarks, face2_landmarks)

# print("두 얼굴의 유사도:", similarity)
