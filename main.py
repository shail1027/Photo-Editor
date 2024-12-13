from ui import ui
import tkinter as tk
from tkinter import messagebox

"""
2023216083 이예빈
영상처리-기말고사 프로젝트 최종 제출물
 
프로젝트의 메인 실행 파일입니다.
"""

# Tkinter로 메인 윈도우 생성
root = tk.Tk()
root.title("영상처리-기말프로젝트")  # 윈도우 타이틀
root.geometry("1100x800")  # 윈도우 크기
ui(root)  # ui.py파일의 ui함수 호출


def closing():
    """창을 닫을 때 호출되는 함수"""
    if messagebox.askokcancel("나가기", "정말 창을 닫으시겠습니까?"):
        root.destroy()  # Tkinter 창을 닫고 프로그램 종료


# 창 닫기(X)를 눌렀을 때 closing함수 호출
root.protocol("WM_DELETE_WINDOW", closing)


root.mainloop()
