import cv2
import numpy as np

# 전역 변수
current_image = None  # 현재 이미지
original_image = None  # 원본 이미지


def set_image(img):  # 현재 이미지& 원본 이미지 설정
    global current_image, original_image
    current_image = img
    original_image = img.copy()  # 원본 이미지 저장


def get_image():  # 현재 이미지 반환
    return current_image


def retro_filter():  # 레트로 필터
    global current_image
    # 누리끼리한 2010년대 필터 느낌
    b, g, r = cv2.split(current_image)  # BRG 채널 분리
    r = cv2.add(r, 50)
    g = cv2.add(g, 30)
    current_image = cv2.merge((b, g, r))  # 채널 병합


def vintage_filter():  # 빈티지 필터
    global current_image
    # 필름노이즈 추가 및 색감조정
    b, g, r = cv2.split(current_image)  # BRG 채널 분리
    r = cv2.add(r, 40)
    g = cv2.add(g, 20)
    b = cv2.add(b, 10)
    current_image = cv2.merge((b, g, r))  # 채널 병합

    # 대비 감소 및 밝기 감소
    current_image = cv2.convertScaleAbs(current_image, alpha=0.8, beta=-10)

    # 따뜻한 갈색 톤 오버레이
    overlay = np.full_like(current_image, (20, 10, 0))  # 약간의 갈색 필터
    current_image = cv2.addWeighted(current_image, 0.9, overlay, 0.1, 0)

    # 필름 노이즈 텍스쳐 추가
    noise_texture = cv2.imread(
        "FilmNoise.png", cv2.IMREAD_COLOR
    )  # 노이즈 텍스처 크기 조정
    noise_texture = cv2.resize(
        noise_texture, (current_image.shape[1], current_image.shape[0])
    )
    # 텍스쳐 합성
    current_image = cv2.addWeighted(current_image, 0.95, noise_texture, 0.3, 0)


def adjust_brightness(n):  # 밝기 조정
    global current_image
    temp = current_image + n
    temp = np.clip(temp, 0, 255).astype(np.uint8) 
    current_image = temp

def adjust_contrast(n):  # 대비 조정
    global current_image
    current_image = cv2.convertScaleAbs(current_image, alpha=n, beta=0)


def adjust_saturation(n):  # 채도 조정
    global current_image
    hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], n)
    current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_hue(hue_shift): # 색조 조정
    global current_image
    hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180  # Hue 범위 [0, 179]
    current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def convert_to_grayscale(): # 흑백화
    global current_image
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)


def sharpen_filter(): # 선명 효과(샤프닝)
    global current_image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    current_image = cv2.filter2D(current_image, -1, kernel)


def edge_detection(): # 윤곽선 추출
    global current_image

    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200) 
    current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) 


def blur_filter(): # 블러 효과
    global current_image
    kernel_size = 5
    current_image = cv2.GaussianBlur(current_image, (kernel_size, kernel_size), 0)


def edge_emphasize(): # 윤곽선 강조(스케치 효과)
    edge = 20 # 밝기 차이의 기준 값
    global current_image
    height, width, _ = current_image.shape # 현재 이미지의 높이, 너비
    gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY) # 밝기 정보만 사용할거기때문에 흑백으로 변환
    temp = np.copy(current_image) # 기존 이미지 복사

    for col in range(height): # 이미지의 픽셀을 순회하며 윤곽선을 계산
        for row in range(width):
            mono = int(gray_image[col, row])

            # 오른쪽과 아래쪽 픽셀을 비교하기때문에 이미지의 경계를 확인
            if row + 1 < width and col + 1 < height: # 만약 경계에 걸리지 않는다면
                right = int(gray_image[col, row + 1]) # 오른쪽 픽셀의 밝기 값
                bottom = int(gray_image[col + 1, row]) # 아래 픽셀의 밝기 값

                diff_right = abs(mono - right) # 현재 픽셀과 오른쪽 픽셀의 밝기 차
                diff_bottom = abs(mono - bottom) # 현재 픽셀과 아래 픽셀의 밝기 차

                if diff_right > edge: # 만약 기준값(edge)보다 크다면 윤곽선으로 간주한다
                        temp[col, row] = [diff_right] * 3
                elif diff_bottom > edge:
                        temp[col, row] = [diff_bottom] * 3
                else: # 윤곽선이 아닐 경우 원래 픽셀 값을 유지
                        temp[col, row] = current_image[col, row]
            else: # 경계 픽셀은 원본 값을 유지
                    temp[col, row] = current_image[col, row]

    current_image = temp  # 결과 이미지를 저장


def original(): # 원본 이미지로 되돌리기
    global current_image, original_image
    current_image = original_image
