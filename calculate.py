# calculate.py (연결 테스트용)
import json

def calculate_ai_score(t0, t1, type='sky'):
    return 38.5  # 두 번 더해서 77점이 되도록 함

def calculate_sal_logic(token0, token1, gender0, gender1, current_score):
    # 무조건 77점으로 고정
    fixed_score = 77.7
    
    # 테스트용 가짜 살 데이터
    sal0 = [1, 0, 0, 0, 0, 0, 0, 0] # 1번 살에 1점
    sal1 = [0, 1, 0, 0, 0, 0, 0, 0] # 2번 살에 1점
    
    return fixed_score, sal0, sal1

# (중요) app.py에서 import 할 때 에러 안 나도록 빈 함수 유지
def load_models():
    return True
