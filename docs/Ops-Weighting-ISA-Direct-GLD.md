# Ops-Weighting-ISA-Direct-GLD  
AURORA Rev12.1b 포트폴리오 비중을 **ISA 165 / 직투 235** 구조에 정확하게 매핑하는 공식 운용 문서  
(한글 base + 영문 혼용)

---

## 1. Purpose  
이 문서는 Rev12.1b 엔진이 산출한 **Target Portfolio Weights**를  
ISA 계좌(165만원)와 Direct 계좌(235만원)에 어떻게 실제로 배분할지를 명확히 정의한다.  

AURORA 엔진은 계좌를 구분하지 않고 **Combined Portfolio 400만원 기준 비중**을 출력한다.  
따라서 이 비중을 **ISA / Direct 로 분해(mapping)** 하는 절차가 필요하다.

---

## 2. Fixed Monthly Deposit Structure  
- **ISA 월 납입 한도: 1,650,000 KRW**  
- **Direct 월 납입: 2,350,000 KRW**  
- **총 합계: 4,000,000 KRW (ISA + Direct)**  

본 문서는 AURORA의 엔진 출력값(Target Weights)을  
이 두 계좌에 정확히 분해하는 절차를 정의한다.

---

## 3. Asset Eligibility Restrictions  
계좌별로 매수 가능한 자산은 다음과 같이 고정된다.

### 3.1 ISA Account (Allowed Assets)
- SPX Proxy: **VOO**  
- NDX Proxy: **QQQ**  
- DIV Proxy: **SCHD**  
- Gold: **GLDM**  
- Bonds: **Not allowed (TLT, EDV, VGSH, SGOV 금지)**

### 3.2 Direct Account (Allowed Assets)
- 모든 자산 매수 가능  
  - VOO / QQQ / SCHD  
  - VWO (EM), XLE (Energy)  
  - SGOV (T-Bill), TLT/EDV (Duration)  
  - GLD (Gold)

---

## 4. Engine Output to Real Accounts Mapping  
엔진이 산출하는 자산군은 다음과 같다.

- **Core Equity**  
  - SPX  
  - NDX  
  - DIV  

- **Satellite** (EM + Energy)  
- **Duration** (0–15%)  
- **SGOV** (0–80%)  
- **Gold** (Fixed at 5% rule unless violated)  

이 비중(합계 100%)은 400만원 기준이며,  
ISA 165만 / Direct 235만으로 나누어야 한다.

---

## 5. Gold 5% Policy (GLD / GLDM)  
Rev12.1b Gold rule:

### 5.1 Combined Portfolio Rule  
- Combined portfolio 기준 Gold target = **5%**  
  (≈ 200,000 KRW per 4,000,000 KRW)

### 5.2 Monthly Contribution Rule  
Gold는 다음 방식으로 월 납입을 고정한다:

- **ISA → GLDM 100,000 KRW**  
- **Direct → GLD 100,000 KRW**

### 5.3 ISA Priority Rule  
Combined Gold 비중이 5% 미만일 경우:  
→ **추가 Gold 매수는 ISA에서만 수행**  

### 5.4 Gold Overweight Rule  
5% 초과 시:  
→ 초과분은 Direct에서 조정 (ISA Gold 패시브 유지)

---

## 6. Core Equity Mapping (SPX / NDX / DIV)  
엔진이 산출하는 Core Equity 합계는: 
Core = 100 – SGOV – Satellite – Duration – Gold(5%)


Core Equity 내부 비율(고정):

- **SPX = 52.5%**  
- **NDX = 24.5%**  
- **DIV = 23.0%**

### 6.1 Account Assignment  
- ISA: VOO / QQQ / SCHD 로 Core 비중 일부 매수  
- Direct: 남는 Core 비중 매수

ISA 우선 매수 원칙:  
- Core 비중은 가능한 한 **ISA에서 먼저 채우고**,  
- 부족분을 Direct에서 채운다.

이유: ISA의 과세 이점.

---

## 7. Satellite Allocation  
Satellite = **EM + Energy**, 내부 비율 = **2 : 1**

- EM = VWO  
- Energy = XLE  
- ISA에서는 Satellite 매수 금지.  
- Direct 계좌에서만 위 비중대로 매수.

---

## 8. Duration Allocation  
Duration (TLT 또는 EDV)  
- ISA = **매수 금지**  
- Direct 계좌에서만 전량 배분

엔진에서 Duration target %가 나오면  
해당 비중은 **Direct에서 100% 처리**한다.

---

## 9. SGOV Floor Allocation  
SGOV는 단기 국채 머니마켓 대체 자산.

- SGOV 역시 ISA 매수 금지  
- Direct 계좌에서만 매수  
- ISA는 SGOV 비중을 0%로 고정 유지  
- Direct에서 엔진 타겟 %를 100% 반영

---

## 10. Final Mapping Formula  
400만원을 기준으로 엔진의 target weights를 받은 후:

### 10.1 ISA Allocation  
ISA_asset_value = 1,650,000 KRW × Weight(ISA-allowed assets only)


### 10.2 Direct Allocation  
Direct_asset_value = 2,350,000 KRW × Weight(all assets)
– ISA_asset_value (for overlapping Core assets)

즉:

- Core Equity 일부는 ISA + Direct 로 나뉘고  
- Satellite, Duration, SGOV 는 Direct only  
- Gold는 고정(ISA 10만 / Direct 10만)

---

## 11. Example (Illustrative Only)

### Engine Output
SGOV = 22%
Satellite = 3%
Duration = 0%
Gold = 5%
Core = 70%

Core Equity 배분:
SPX = 36.75%
NDX = 17.15%
DIV = 16.10%

ISA 배분:
- Core 일부 + Gold 10만

Direct 배분:
- Satellite 3% + SGOV 22% + 귀속 Core + Gold 10만

---

## 12. Summary
이 문서는 엔진이 산출한 **Target Weights(100%)**를  
ISA 165 + Direct 235 구조에 정확히 매핑하기 위한  
**공식 비중 규정 및 운용 절차**를 정의한다.

ISA: Core / Gold 중심  
Direct: Satellite / Duration / SGOV 중심  
Gold: 10만 + 10만 (필요 시 ISA 추가 매수)

이 규정을 기준으로 모든 계좌 배분을 일관되게 처리한다.

---
