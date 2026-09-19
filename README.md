# Shooting Form Analysis

농구 슛 영상을 보고 “누구랑 비슷해 보인다” 정도에서 끝내지 않고, **관절 움직임 전체를 같은 기준으로 맞춘 뒤 비교해보려고 만든 실험용 분석기**입니다.

Flask 웹앱에서 1~3개 시점의 영상을 올리고, MediaPipe pose를 이용해 catch부터 follow-through까지의 움직임을 비교합니다.

> 이 프로젝트는 코칭 보조 실험입니다. 의료·재활·정밀 생체역학 측정 도구가 아닙니다.

## 분석 알고리즘

### 1. Pose 추출

```text
영상 업로드
   ↓
분석할 사람 선택
   ↓
각 frame에서 MediaPipe 33 landmarks 추출
   ↓
유효한 pose sequence 구성
```

### 2. 체형 영향 줄이기

단순 픽셀 좌표를 그대로 비교하면 키나 카메라 거리 때문에 결과가 흔들립니다.

```text
33 landmarks
   ↓
몸통을 기준으로 중심 정렬
   ↓
torso scale로 크기 정규화
   ↓
오른손 / 왼손 shooting side 정렬
```

이렇게 하면 팔이 실제로 더 긴 사람이라는 이유만으로 비슷하거나 다르다고 판정되는 영향을 줄일 수 있습니다.

### 3. 슛 시간축 맞추기

두 사람이 슛을 같은 속도로 하지 않기 때문에 frame 번호끼리 바로 비교하지 않습니다.

```text
catch → set → release → follow-through phase 검출
        ↓
phase 경계를 기준으로 sequence 정렬
        ↓
phase-aligned DTW
        ↓
서로 대응되는 동작 frame 탐색
        ↓
전체 움직임 distance 계산
```

DTW(Dynamic Time Warping)는 한 사람이 조금 빨리 움직이고 다른 사람이 조금 느리게 움직여도 비슷한 동작 구간끼리 대응시킬 수 있게 해줍니다.

### 4. 선수 비교

```text
사용자 normalized motion
        ↓
각 player profile과 같은 방식으로 DTW distance 계산
        ↓
distance가 작은 순서로 비교
        ↓
선택한 선수와의 상세 비교 + nearest player 표시
```

시각화에서는 선수와 사용자의 motion을 같은 일반 성인 skeleton 비율로 retarget해서 **각도와 움직임 차이**에 집중합니다.

## 기능

1. side / front / oblique 1~3개 시점 업로드
2. 영상 속 분석 대상 선택
3. 33 landmark 정규화
4. phase-aligned DTW 비교
5. 선택 선수 비교와 nearest-player 탐색
6. 공통 skeleton으로 motion retarget
7. 3D timeline 회전·확대·scrub·재생

## 실행

```bash
pip install -r requirements.txt
python run.py
# http://127.0.0.1:7860
```

검증:

```bash
python -m unittest discover -s tests -v
python scripts/validate_motion_dataset.py --min-clips 3
```

## 데이터 처리

선수 모델은 품질 필터를 통과한 공개 슈팅 영상에서 학습용 motion profile을 만듭니다. 저장소에는 candidate metadata와 source URL만 남기고, 다운로드한 원본 영상 자체를 재배포하지 않습니다.

데이터를 다시 만들려면:

```bash
pip install -r requirements-data.txt
python scripts/build_youtube_catalog.py
python scripts/discover_allstar_players.py
```

## 구조

| 경로 | 역할 |
|---|---|
| `app/` | pose, 각도, similarity, DB, Flask API |
| `static/` | 웹 UI |
| `models/` | motion profile, source catalog, validation report |
| `scripts/` | 데이터 탐색·학습·검증 |
| `data/` | runtime SQLite |
| `Dockerfile` | production image |
| `design-system/` | UI 규칙 |

## 한계

- 카메라 각도, framing, 가림에 영향을 받습니다.
- monocular landmark depth는 실제 계측된 3D가 아닙니다.
- 고정 skeleton은 선수의 실제 신체 치수를 재현하지 않습니다.
- player profile은 공개 영상에서 계산한 비공식 분석값이며 선수·리그와 제휴된 데이터가 아닙니다.
- 특이한 편집 영상에서는 release 구간 검출이 틀릴 수 있습니다.

Render 배포 설정은 `DEPLOYMENT.md`에 있습니다.
