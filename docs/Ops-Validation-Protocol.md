# AURORA Rev12.1b – Cross-Validation Protocol (ChatGPT ↔ Grok)

## Purpose
동일한 시장데이터 JSON과 동일한 엔진 코드를 입력할 때,
ChatGPT와 Grok이 **동일한 FD → ML → Systemic → State → Target Weights** 를 출력하는지 검증.

## Procedure
1) 다음 명령어를 Grok과 ChatGPT에 각각 입력한다:

오늘 날짜 AURORA Rev12.1b 엔진 구동.
GitHub 레포:
https://github.com/paulhjung-gh/AURORA-Rev12.1b-Core

market_data_20251210.json 파일 기준으로 엔진 구동.
FD → ML → Systemic → State → SGOV/Satellite/Duration/Core 비중 계산.

2) 두 모델이 출력한 값 비교:
- FXW
- MacroScore
- Drawdown
- FXVol
- ML_Risk / ML_Opp / ML_Regime
- Systemic Level
- State 
- SGOV / Satellite / Duration / Core 비중

3) 오차 기준:
- 수식 기반이므로 오차 허용 범위 = ±0.00  
- 둘 중 하나가 다른 값을 내면 **해석 문제**로 간주하고 실행 로그 확인

