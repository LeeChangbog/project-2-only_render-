from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
from calculate import load_models, calculate_ai_score, calculate_sal_logic

# Flask 앱 생성
app = Flask(__name__)
# 모든 도메인(Netlify 등)에서의 요청 허용
CORS(app)

# 서버 시작 시 모델 로드 시도
print(">>> Server Starting... Loading Models...")
load_success = load_models()
if load_success:
    print(">>> Models are ready.")
else:
    print(">>> WARNING: Models failed to load.")

@app.route('/')
def home():
    return "Saju AI Server is Running on Render!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. 데이터 받기
        data = request.json
        token0 = data.get('token0') # [년천, 년지, 월천, 월지, 일천, 일지, 시천, 시지]
        token1 = data.get('token1')
        gender0 = int(data.get('gender0', 1))
        gender1 = int(data.get('gender1', 2))

        # 2. AI 점수 계산 (천간 + 지지)
        # 사주 궁합의 핵심인 '일간(Day Sky)'과 '일지(Day Earth)'를 기준으로 계산
        # 인덱스: 년(0,1), 월(2,3), 일(4,5), 시(6,7)
        
        # 일간 궁합 (Index 4) -> 40~60점 만점 기준 보정
        sky_score_raw = calculate_ai_score(token0[4], token1[4], type='sky')
        
        # 일지 궁합 (Index 5) -> 40~60점 만점 기준 보정
        earth_score_raw = calculate_ai_score(token0[5], token1[5], type='earth')
        
        # 단순 합산 (기본 점수)
        base_score = sky_score_raw + earth_score_raw
        
        # 점수 범위 안전장치 (혹시라도 너무 크거나 작으면 보정)
        if base_score > 100: base_score = 100
        if base_score < 60: base_score = 60 # 기본 궁합 최저점 보정
        
        # 3. 살(Sal) 계산 및 감점
        final_score, sal0, sal1 = calculate_sal_logic(token0, token1, gender0, gender1, base_score)

        # 4. 결과 반환
        return jsonify({
            "score": round(final_score, 1),
            "sal0": sal0,
            "sal1": sal1
        })

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 로컬 테스트용
    app.run(host='0.0.0.0', port=5000)