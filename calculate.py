import numpy as np
import sys
import os

# ---------------------------------------------------------
# 1. TensorFlow 및 모델 설정
# ---------------------------------------------------------
model_sky = None
model_earth = None

def load_models():
    """
    서버가 시작될 때 .h5 모델 파일들을 메모리에 로드합니다.
    Render 서버에 sky3000.h5, earth3000.h5 파일이 같이 있어야 합니다.
    """
    global model_sky, model_earth
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.metrics import MeanSquaredError
        
        # 파일 경로 확인 (현재 폴더 기준)
        base_path = os.path.dirname(os.path.abspath(__file__))
        sky_path = os.path.join(base_path, 'sky3000.h5')
        earth_path = os.path.join(base_path, 'earth3000.h5')

        print(f"Loading models from: {sky_path}, {earth_path}")
        
        # 커스텀 객체(mse)와 함께 로드
        model_sky = load_model(sky_path, custom_objects={'mse': MeanSquaredError})
        model_earth = load_model(earth_path, custom_objects={'mse': MeanSquaredError})
        print(">>> AI Models loaded successfully!")
        return True
    except Exception as e:
        print(f">>> Error loading models: {e}")
        return False

def calculate_ai_score(t0_val, t1_val, type='sky'):
    """
    AI 모델을 사용하여 두 글자(숫자) 사이의 궁합 점수를 예측합니다.
    type: 'sky' (천간, 10개), 'earth' (지지, 12개)
    """
    global model_sky, model_earth
    
    # 모델이 로드되지 않았으면 기본값 반환
    if (type == 'sky' and model_sky is None) or (type == 'earth' and model_earth is None):
        return 50.0

    try:
        nb_classes = 10 if type == 'sky' else 12
        model = model_sky if type == 'sky' else model_earth
        
        # 원-핫 인코딩 생성 (입력값이 1부터 시작하므로 -1 해줌)
        idx0 = int(t0_val) - 1
        idx1 = int(t1_val) - 1
        
        # 범위 체크
        if idx0 < 0: idx0 = 0
        if idx1 < 0: idx1 = 0
        if idx0 >= nb_classes: idx0 = nb_classes - 1
        if idx1 >= nb_classes: idx1 = nb_classes - 1

        vec0 = np.eye(nb_classes)[np.array(idx0).reshape(-1)].flatten()
        vec1 = np.eye(nb_classes)[np.array(idx1).reshape(-1)].flatten()
        
        # 두 벡터를 합쳐서 모델에 입력
        input_vec = np.concatenate((vec0, vec1)).reshape(1, -1)
        
        # 예측 (결과는 [[score]] 형태)
        prediction = model.predict(input_vec, verbose=0)
        return float(prediction[0][0])
        
    except Exception as e:
        print(f"AI Prediction Error: {e}")
        return 50.0

# ---------------------------------------------------------
# 2. 살(Sal) 계산 파라미터
# ---------------------------------------------------------
p1, p11 = 8, 9.5
p2, p21 = 7, 8.2
p3, p31 = 6, 7.2
p41, p42, p43 = 10, 8, 6
p5 = 8
p6 = 8
p7, p71 = 0, 10
p8, p81, p82, p83 = 0, 10, 6, 4

# ---------------------------------------------------------
# 3. 살(Sal) 계산 로직 (독립적 누적 방식)
# ---------------------------------------------------------
def calculate_sal_logic(token0, token1, gender0, gender1, current_score):
    
    # 입력값 정수 변환
    a1, a2, a3 = int(token0[1]), int(token0[3]), int(token0[5])
    b1, b2, b3 = int(token1[1]), int(token1[3]), int(token1[5])
    
    t0_sky_year, t0_sky_month, t0_sky_day = int(token0[0]), int(token0[2]), int(token0[4])
    t1_sky_year, t1_sky_month, t1_sky_day = int(token1[0]), int(token1[2]), int(token1[4])

    sal0 = [0] * 8
    sal1 = [0] * 8

    # [1] 삼형살 등 (p1)
    if a3 == 3 and (a1 in [6,9] or a2 in [6,9]):
        val = p1 if gender0 == 1 else p11
        current_score -= val; sal0[0] += val
    if a3 == 7 and (a1 in [2,5,7] or a2 in [2,5,7]):
        val = p1 if gender0 == 1 else p11
        current_score -= val; sal0[0] += val
    if a3 == 2 and (a1 in [7,8,11] or a2 in [7,8,11]):
        val = p1 if gender0 == 1 else p11
        current_score -= val; sal0[0] += val
        
    if b3 == 3 and (b1 in [6,9] or b2 in [6,9]):
        val = p1 if gender1 == 1 else p11
        current_score -= val; sal1[0] += val
    if b3 == 7 and (b1 in [2,5,7] or b2 in [2,5,7]):
        val = p1 if gender1 == 1 else p11
        current_score -= val; sal1[0] += val
    if b3 == 2 and (b1 in [7,8,11] or b2 in [7,8,11]):
        val = p1 if gender1 == 1 else p11
        current_score -= val; sal1[0] += val

    # [2] 원진살 (p2)
    val0_p2 = p2 if gender0 == 1 else p21
    val1_p2 = p2 if gender1 == 1 else p21
    wonjin_pairs = [(1,8), (2,7), (3,10), (4,9), (5,12), (6,11), 
                    (8,1), (7,2), (10,3), (9,4), (12,5), (11,6)]
    
    if (a3, a1) in wonjin_pairs or (a3, a2) in wonjin_pairs:
        current_score -= val0_p2; sal0[1] += val0_p2
    if (b3, b1) in wonjin_pairs or (b3, b2) in wonjin_pairs:
        current_score -= val1_p2; sal1[1] += val1_p2

    special_relations = [(1,10), (2,7), (3,8), (4,9), (5,12), (6,11), (10,1), (7,2), (8,3), (9,4), (12,5), (11,6)]
    if (a1, a2) in special_relations: current_score -= val0_p2; sal0[1] += val0_p2
    if (b1, b2) in special_relations: current_score -= val1_p2; sal1[1] += val1_p2

    # [3] 형살 (p3)
    if (a1==1 and a2==4) or (a1==1 and a3==4) or (a2==1 and a3==4) or \
       (a1==4 and a2==1) or (a1==4 and a3==1) or (a2==4 and a3==1):
       current_score -= p3; sal0[2] += p3
    if (b1==1 and b2==4) or (b1==1 and b3==4) or (b2==1 and b3==4) or \
       (b1==4 and b2==1) or (b1==4 and b3==1) or (b2==4 and b3==1):
       current_score -= p3; sal1[2] += p3

    self_hyeong = [5, 7, 10, 12]
    if a1 in self_hyeong and (a2 == a1 or a3 == a1): current_score -= p3; sal0[2] += p3
    if a2 in self_hyeong and a3 == a2: current_score -= p3; sal0[2] += p3
    if b1 in self_hyeong and (b2 == b1 or b3 == b1): current_score -= p3; sal1[2] += p3
    if b2 in self_hyeong and b3 == b2: current_score -= p3; sal1[2] += p3

    # [4] 충살 (p4)
    if abs(a3 - a2) == 6: current_score -= p41; sal0[3] += p41
    if abs(a3 - a1) == 6: current_score -= p42; sal0[3] += p42
    if abs(a1 - a2) == 6: current_score -= p43; sal0[3] += p43
    
    if abs(b3 - b2) == 6: current_score -= p41; sal1[3] += p41
    if abs(b3 - b1) == 6: current_score -= p42; sal1[3] += p42
    if abs(b1 - b2) == 6: current_score -= p43; sal1[3] += p43

    # [5] 파살 (p5)
    pa_pairs = [(1,10), (10,1), (2,5), (5,2), (3,12), (12,3), (4,7), (7,4), (6,9), (9,6), (11,8), (8,11)]
    if (a3, a2) in pa_pairs or (a3, a1) in pa_pairs: current_score -= p5; sal0[4] += p5
    if (b3, b2) in pa_pairs or (b3, b1) in pa_pairs: current_score -= p5; sal1[4] += p5

    # [6] 해살 (p6)
    hae_pairs = [(1,8), (8,1), (2,7), (7,2), (3,6), (6,3), (4,5), (5,4), (9,12), (12,9), (10,11), (11,10)]
    if (a3, a2) in hae_pairs or (a3, a1) in hae_pairs: current_score -= p6; sal0[5] += p6
    if (b3, b2) in hae_pairs or (b3, b1) in hae_pairs: current_score -= p6; sal1[5] += p6

    # [7] 백호/괴강 (p7)
    special_star = [(5,5), (4,2), (3,11), (2,8), (1,5), (10,2), (9,11)]
    if (t0_sky_day, a3) in special_star:
        val = p7 if gender0==1 else p71
        current_score -= val; sal0[6] += val
    if (t1_sky_day, b3) in special_star:
        val = p7 if gender1==1 else p71
        current_score -= val; sal1[6] += val

    # [8] 성별 특수 (p8)
    bad_set = [(9,5), (5,11), (7,5), (7,11)]
    if gender0 != 1:
        if (t0_sky_day, a3) in bad_set: current_score-=p81; sal0[7]+=p81
        if (t0_sky_month, a2) in bad_set: current_score-=p82; sal0[7]+=p82
        if (t0_sky_year, a1) in bad_set: current_score-=p83; sal0[7]+=p83

    if gender1 != 1:
        if (t1_sky_day, b3) in bad_set: current_score-=p81; sal1[7]+=p81
        if (t1_sky_month, b2) in bad_set: current_score-=p82; sal1[7]+=p82
        if (t1_sky_year, b1) in bad_set: current_score-=p83; sal1[7]+=p83

    return current_score, sal0, sal1
