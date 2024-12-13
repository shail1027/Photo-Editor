import cv2
import numpy as np
from PIL import Image, ImageTk

# 전역 변수
current_image = None  # 현재 이미지
original_image = None  # 원본 이미지

undo_stack = []  # 이전 상태 저장 스택
redo_stack = []  # 되돌린 작업 다시 실행을 위한 스택


# ------ 이미지 초기화 ------#
def init_image(img):
    """현재 이미지, 원본 이미지 초기화"""
    global current_image, original_image
    current_image = img
    original_image = img.copy()  # 원본 이미지 저장


def set_image(img):
    """현재 이미지 설정"""
    global current_image
    current_image = img


def get_image():
    """현재 이미지 반환"""
    return current_image


def original():  # 원본 이미지로 되돌리기
    global current_image, original_image
    current_image = original_image


# ------Ctrl+z, Ctrl+y관련 처리 ------#
def push_to_undo(img):
    """현재 상태를 undo 스택에 저장"""
    global undo_stack
    if img is not None:
        undo_stack.append(img.copy())


def pop_from_undo():
    """undo 스택에서 상태를 가져옴"""
    global undo_stack
    if undo_stack:
        return undo_stack.pop()
    return None


def push_to_redo(img):
    """현재 상태를 redo 스택에 저장"""
    global redo_stack
    if img is not None:
        redo_stack.append(img.copy())


def pop_from_redo():
    """redo 스택에서 상태를 가져옴"""
    global redo_stack
    if redo_stack:
        return redo_stack.pop()
    return None


# ------ 대비 조정 ------#
def adjust_contrast(n, beta=0):
    """대비 조정 함수"""

    global current_image

    if current_image is None:
        print("Error: No image.")
        return

    try:
        # 대비 조정
        temp = cv2.convertScaleAbs(current_image, alpha=n, beta=beta)

        # 값이 0~255 범위를 벗어나지 않도록 자동 처리
        current_image = np.clip(temp, 0, 255).astype(np.uint8)

    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


# ------ 채도 조정 ------#
def adjust_saturation(n):
    """채도 조정 함수"""
    global current_image

    if current_image is None:
        print("Error: No image.")
        return

    try:
        # 이미지 HSV 변환
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)

        # 채도 조정
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], n)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)  # 채도 값은 0~255 범위 유지

        # HSV를 BGR로 변환하여 이미지 업데이트
        current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"Error : {e}")


# ------ 색조 조정 ------#
def adjust_hue(hue_shift):
    """색조 조정 함수"""
    global current_image

    if current_image is None:
        print("Error: No image.")
        return

    try:
        # 이미지 HSV 변환
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)

        # Hue 범위 검증 및 조정
        hue_shift = hue_shift % 180  # Hue 범위는 0 ~ 179
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

        # HSV를 BGR로 변환하여 이미지 업데이트
        current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"Error : {e}")


# ------ 이미지 흑백화 ------#
def convert_to_grayscale():
    """이미지 흑백화"""
    global current_image
    if current_image is None:
        print("Error: No image.")
        return

    # 밝기(Grayscale)를 계산
    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    # Grayscale을 3채널로 변환하여 색상 정보 유지
    current_image = cv2.merge([gray, gray, gray])


# ------ 윤곽선 추출 ------#
def edge_detection():
    """이미지 윤곽선 추출"""
    global current_image

    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


# ------ 블러 적용 ------#
def apply_blur(image, start_point, end_point, radius):
    """블러 효과를 적용하는"""
    global current_image
    x, y = end_point
    print(radius)
    # 이미지 크기 가져오기
    h, w = image.shape[:2]

    # ROI 경계 설정
    x_start, x_end = max(0, x - radius), min(w, x + radius)
    y_start, y_end = max(0, y - radius), min(h, y + radius)

    # ROI 추출
    roi = image[y_start:y_end, x_start:x_end].copy()

    # 블러링 필터 적용
    blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)

    # 원본 이미지에 반영
    image[y_start:y_end, x_start:x_end] = blurred_roi
    current_image = image


# ------ 선명 효과 적용 ------#
def sharpen_filter():
    """선명 효과(샤프닝) 적용"""
    global current_image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    current_image = cv2.filter2D(current_image, -1, kernel)


# ------ 레트로 필터 적용 ------#
def retro_filter():
    """레트로 느낌의 필터 적용"""
    global current_image
    # 누리끼리한 2010년대 필터 느낌
    b, g, r = cv2.split(current_image)  # BRG 채널 분리
    r = cv2.add(r, 50)
    g = cv2.add(g, 30)
    current_image = cv2.merge((b, g, r))  # 채널 병합


# ------ 잡티제거 효과 적용 ------#
def remove_noise(image, center):
    """
    이미지에서 잡티를 제거
    :param image: 처리할 Resized 이미지
    :param center: 클릭한 중심 좌표 (x, y)
    """
    global current_image
    try:
        # 중심 좌표와 커널 크기 설정
        x, y = center
        kernel_size = 7  # 커널 크기
        half_k = kernel_size // 2

        # 이미지 크기 가져오기
        h, w = image.shape[:2]

        # ROI 경계 설정
        x_start = max(0, x - half_k)
        y_start = max(0, y - half_k)
        x_end = min(w, x + half_k + 1)
        y_end = min(h, y + half_k + 1)

        # ROI 추출
        roi = image[y_start:y_end, x_start:x_end]

        # 중간값 필터 적용
        filtered_roi = cv2.medianBlur(roi, kernel_size)

        # 필터링된 결과를 Resized 이미지에 반영
        image[y_start:y_end, x_start:x_end] = filtered_roi
        current_image = image

    except Exception as e:
        print(f"Error: {e}")



# ------ y2k필터 적용 ------#
def y2k_filter():
    """
    y2k필터 적용
    1. 대비 +40
    2. 채도 -20
    3. 모션 블러 적용
    """
    global current_image

    if current_image is None:
        print("Error: No image.")
        return

    try:
        # Step 1: 대비 +60
        contrast = 1.4  # 대비 값 (1.0 = 기본값, 1.6 = 60% 증가)
        beta = 0  # 밝기 보정 값
        current_image = cv2.convertScaleAbs(current_image, alpha=contrast, beta=beta)

        # Step 2: 채도 -20
        saturation_adjustment = 0.8  # 채도 비율 (1.0 = 기본값, 0.8 = 20% 감소)
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_adjustment, 0, 255).astype(
            np.uint8
        )
        current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Step 3: 모션 블러 적용
        kernel_size = 7  # 모션 블러 커널 크기
        motion_blur_kernel = np.zeros((kernel_size, kernel_size)) 
        np.fill_diagonal(motion_blur_kernel, 1)  # 대각선 방향으로 값 설정
        motion_blur_kernel /= kernel_size  # 정규화
        current_image = cv2.filter2D(current_image, -1, motion_blur_kernel)

    except Exception as e:
        print(f"Error applying filters: {e}")


# ------ 무채색 필터 적용 ------#
def mono_filter():
    """
    무채색 필터 적용
    휘도 -20, 하이라이트 -40, 대비 +60, 채도 -30, 색 선명도 +15, 필터 (모노 or 느와르) +40
    """
    global current_image

    if current_image is None:
        print("Error: No image.")
        return

    try:
        # Step 1: 휘도 조정 (-20)
        brightness_adjustment = -20
        temp = current_image.astype(np.int16) + brightness_adjustment
        temp = np.clip(temp, 0, 255).astype(np.uint8)
        current_image = temp

        # Step 2: 하이라이트 조정 (-40)
        highlight_adjustment = -40
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2].astype(np.int16) + highlight_adjustment, 0, 255
        ).astype(np.uint8)
        current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Step 3: 대비 조정 (+60)
        contrast_adjustment = 1.6  # 대비 조정 값 (alpha)
        beta = 0  # 밝기 보정 값
        current_image = cv2.convertScaleAbs(
            current_image, alpha=contrast_adjustment, beta=beta
        )

        # Step 4: 채도 조정 (-30)
        saturation_adjustment = 0.7  # 채도 조정 비율
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_adjustment, 0, 255).astype(
            np.uint8
        )
        current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Step 5: 색 선명도 (+15)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        current_image = cv2.filter2D(current_image, -1, sharpen_kernel)

    except Exception as e:
        print(f"Error applying filter: {e}")


# ------ 윤곽선 강조 효과 적용 ------#
def edge_emphasize():
    """윤곽선 강조 효과(스케치 하듯)"""
    edge = 20  # 밝기 차이의 기준 값
    global current_image
    height, width, _ = current_image.shape  # 현재 이미지의 높이, 너비
    gray_image = cv2.cvtColor(
        current_image, cv2.COLOR_BGR2GRAY
    )  # 밝기 정보만 사용할거기때문에 흑백으로 변환
    temp = np.copy(current_image)  # 기존 이미지 복사

    for col in range(height):  # 이미지의 픽셀을 순회하며 윤곽선을 계산
        for row in range(width):
            mono = int(gray_image[col, row])

            # 오른쪽과 아래쪽 픽셀을 비교하기때문에 이미지의 경계를 확인
            if row + 1 < width and col + 1 < height:  # 만약 경계에 걸리지 않는다면
                right = int(gray_image[col, row + 1])  # 오른쪽 픽셀의 밝기 값
                bottom = int(gray_image[col + 1, row])  # 아래 픽셀의 밝기 값

                diff_right = abs(mono - right)  # 현재 픽셀과 오른쪽 픽셀의 밝기 차
                diff_bottom = abs(mono - bottom)  # 현재 픽셀과 아래 픽셀의 밝기 차

                if (
                    diff_right > edge
                ):  # 만약 기준값(edge)보다 크다면 윤곽선으로 간주한다
                    temp[col, row] = [diff_right] * 3
                elif diff_bottom > edge:
                    temp[col, row] = [diff_bottom] * 3
                else:  # 윤곽선이 아닐 경우 원래 픽셀 값을 유지
                    temp[col, row] = current_image[col, row]
            else:  # 경계 픽셀은 원본 값을 유지
                temp[col, row] = current_image[col, row]

    current_image = temp  # 결과 이미지를 저장


# ------ 픽셀 유동화 적용 ------#
def liquify_pixels(img, start_point, end_point, strength=50, radius=20):
    """픽셀 유동화 적용"""
    global current_image
    h, w = img.shape[:2]

    # 드래그된 방향 계산
    dx, dy = end_point[0] - start_point[0], end_point[1] - start_point[1]

    # 이미지 복사 (변형을 적용할 임시 이미지)
    output = img.copy()

    # 유동화 영역을 위한 반복문
    for y in range(max(0, start_point[1] - radius), min(h, start_point[1] + radius)):
        for x in range(
            max(0, start_point[0] - radius), min(w, start_point[0] + radius)
        ):
            distance = np.sqrt((x - start_point[0]) ** 2 + (y - start_point[1]) ** 2)

            # 반지름 내의 픽셀에 대해서만 유동화 적용
            if distance < radius:
                ratio = (radius - distance) / radius  # 반지름 내 픽셀의 비율
                new_x = int(x + dx * ratio * strength / 100)  # 새로운 x 좌표
                new_y = int(y + dy * ratio * strength / 100)  # 새로운 y 좌표

                # 이미지 크기를 벗어나지 않도록 클리핑
                new_x = np.clip(new_x, 0, w - 1)
                new_y = np.clip(new_y, 0, h - 1)

                # 새로운 위치로 픽셀 이동
                output[y, x] = img[new_y, new_x]

    current_image = output  # 결과 이미지 갱신


# ------ 메이크업 브러쉬 적용 ------#
def apply_makeup(
    image, start_point, end_point, color=(0, 0, 255), size=20, intensity=0.01
):
    """특정 영역의 색상을 강조 (브러시 스타일) """
    global current_image
    try:
        x, y = end_point
        radius = size  # 브러시 반지름을 size로 설정

        # 이미지 크기 가져오기
        h, w = image.shape[:2]

        # ROI 경계 설정
        x_start, x_end = max(0, x - radius), min(w, x + radius)
        y_start, y_end = max(0, y - radius), min(h, y + radius)

        # ROI 추출
        roi = image[y_start:y_end, x_start:x_end].copy()

        # 브러시 효과를 위한 마스크 생성
        mask = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
        center = (roi.shape[1] // 2, roi.shape[0] // 2)
        cv2.circle(mask, center, radius, color, -1)  # 브러시 색상 적용

        # 마스크를 부드럽게 처리 (경계 부드럽게)
        mask = cv2.GaussianBlur(mask, (61, 61), sigmaX=50)

        # 알파 블렌딩으로 기존 이미지와 혼합
        mask_float = mask.astype(np.float32) / 255.0  # 0~1 범위로 정규화
        blended_roi = (
            roi.astype(np.float32) * (1 - intensity) + mask_float * intensity * 255
        )

        # 원본 이미지에 반영
        image[y_start:y_end, x_start:x_end] = blended_roi.astype(np.uint8)

        current_image = image

    except Exception as e:
        print(f"Error : {e}")
