# 검증된 농구 슛 3D 캡처 및 참조 데이터 운영 가이드

이 문서는 `shooting-form-analysis`에서 **실제 선수의 검증된 3D 모델**과 **승인된 2D 스타일 참조**를 구분해 만드는 절차를 정의합니다. 현재 저장소는 의도적으로 기존 선수 프로필을 `unverified_legacy` 상태로 격리합니다. 따라서 출처·신원·슛 이벤트·검토 기록을 모두 통과하기 전에는 어떤 선수도 매칭이나 3D 스켈레톤 출력에 사용되지 않습니다.

> **핵심 규칙:** 단일 영상에서 만든 pose 시각화는 3D motion capture가 아닙니다. 검증된 3D에는 동일 슛의 보정·동기화된 다중 시점 영상과 재투영 품질 검증이 필요합니다.

## 1. 제품 등급과 공개 조건

| 상태 | 조건 | 허용되는 제품 동작 |
|---|---|---|
| `draft` | 신규 선수 또는 검토 전 source | 관리자 검토만 가능 |
| `unverified_legacy` | 과거 자동 수집 프로필 또는 provenance 부족 | 화면에 상태만 표시, 매칭·스켈레톤 출력 금지 |
| `verified_2d` | 서로 다른 승인 source 3개 이상, 실제 영상·신원·슛·프레임 라벨 확인 | 2D angle/style 참조 매칭 가능 |
| `verified_3d` | `verified_2d` 조건과 canonical calibrated 3D 품질 gate 통과 | 3D 스켈레톤·3D 비교 가능 |
| `rejected` | synthetic, 신원 오류, 비슛 동작, 보정/품질 실패 | 공개·매칭·3D 출력 금지 |

## 2. Curry reference 데이터 계약

Curry의 영상은 시스템의 **스타일 참조**가 될 수 있지만, 동기화된 다중 카메라로 촬영된 검증 자료가 아니라면 Curry의 실제 3D motion capture라고 부르면 안 됩니다. 그 경우 결과는 `verified_2d`까지만 승격할 수 있습니다.

각 클립은 JSON Lines manifest의 한 줄로 기록합니다. `source_url`, 원본 hash, 실제 촬영 여부, 신원 검토, 슛 검토, 사람 검토자, 그리고 catch–release–follow-through frame이 빠지면 클립은 자동 거절됩니다.

```json
{
  "clip_id": "curry_catch_shoot_001",
  "player_key": "stephen_curry",
  "source_url": "https://approved-source.example/video",
  "footage_type": "real",
  "identity_status": "verified",
  "shot_status": "verified",
  "review_status": "approved",
  "reviewer": "reviewer-id",
  "catch_frame": 128,
  "release_frame": 146,
  "followthrough_end_frame": 178,
  "fps": 60,
  "view": "side",
  "license_status": "reviewed",
  "ball_visible_ratio": 0.90,
  "occlusion_ratio": 0.05,
  "notes": "catch-and-shoot; shooter visible for full shot"
}
```

이 manifest는 `POST /api/reference-clips`로 등록할 수 있습니다. API는 clip 계약을 검사하고 해당 선수의 `verification_status`를 다시 계산합니다. 세 개의 **서로 다른 URL**이 같은 선수의 승인된 실제 슛임을 충족해야 `verified_2d`가 됩니다.

## 3. controlled pilot 3D 촬영

시스템의 3D 정확도를 검증하려면 Curry 영상과 분리된 controlled pilot이 필요합니다. 실제 촬영 동의를 받은 참가자 1명이면 충분하며, 목적은 선수 신원이 아니라 3D 파이프라인 검증입니다.

| 항목 | 최소 기준 | 권장 기준 |
|---|---:|---:|
| 카메라 | 고정된 스마트폰 3대 | 고정 카메라 4대 이상 |
| 프레임레이트 | 60 fps | 120 fps |
| 해상도 | 1080p | 4K 또는 고비트레이트 1080p |
| 배치 | 정면, 45도, 측면 | 가림 대응을 위한 후면 사선 추가 |
| 동기화 | 시작 clap 또는 LED flash | 공통 timestamp와 audio/LED 동시 기록 |
| 보정 | 매 세션 checkerboard | Charuco + 재투영 결과 보관 |
| 시도 횟수 | 한 슛 유형 20회 | 슛 유형별 30회 이상 |

카메라는 보정 후 한 번도 움직이면 안 됩니다. 모든 영상에는 전신, 슈팅 팔, 그리고 공이 보여야 하며, 주요 관절은 최소 두 카메라에서 관찰되어야 합니다. OpenCap은 다중 스마트폰 영상, 보정, 동기화, inverse kinematics를 결합하여 사람이 읽을 수 있는 motion output을 만드는 공개 사례입니다.[1] 농구 markerless 연구도 60 fps 이상 다중 카메라, calibration, 수동 스포츠 라벨, 재투영 검증을 결합합니다.[2]

## 4. Calibration 및 synchronization payload

`/api/analyze`에 다중 시점 3D 경로를 요청하려면 multipart form에 `calibration_json`과 `sync_json`을 추가합니다. 각 카메라의 intrinsic, rotation, translation, image size, calibration reprojection error가 필요합니다. 사전 정의된 `front`/`side` yaw나 거리 추정값은 인정되지 않습니다.

```json
{
  "capture_id": "pilot-2026-001",
  "version": "calibration_v1",
  "cameras": {
    "side": {
      "intrinsic": [[1200,0,960],[0,1200,540],[0,0,1]],
      "rotation": [[1,0,0],[0,1,0],[0,0,1]],
      "translation": [0,0,0],
      "image_width": 1920,
      "image_height": 1080,
      "reprojection_error_px": 0.8
    }
  }
}
```

```json
{
  "offsets": {
    "side": {"offset_frames": 0, "confidence": 0.98, "method": "audio_clap"},
    "front": {"offset_frames": 1, "confidence": 0.96, "method": "audio_clap"}
  }
}
```

현재 기본 gate는 camera calibration error 2.5 px 이하, camera offset 2 frame 이하, sync confidence 0.80 이상입니다. 이 값은 file럿 결과가 쌓이면 holdout 데이터에서 조정해야 합니다. 검증에 실패하면 API는 단일 영상 pose 결과를 반환할 수는 있어도, `calibrated_multi_view_3d` 결과를 선언하지 않습니다.

## 5. 슛 이벤트 gate

기존의 “손목이 가장 높은 frame”은 release 후보를 찾는 보조 신호일 뿐입니다. 현재 분석은 다음 증거를 함께 반환합니다.

| 증거 | 현재 baseline | gate 실패 시 동작 |
|---|---|---|
| 선수 연속성 | 선택된 pose bbox의 연속성 | `shooter track is discontinuous`로 거절 |
| 공 후보 | 주황색·원형 공의 보수적 baseline detector | 공이 불충분하면 pose-only |
| pre-release | 공이 슈팅 손 근처인지 | pose-only |
| post-release | 손-공 분리와 상향 공 궤적 | pose-only |
| 다중 시점 3D | 위 증거가 확인된 두 시점 이상 | calibration·sync·reprojection gate로 진행 |

공 detector는 dependency-free baseline이며, court/ball 조명·색상·motion blur에 따라 실패할 수 있습니다. 실패는 잘못된 슛 확정보다 안전하므로, `pose_only_unverified` 사유를 사용자에게 반환합니다. 배포 전에 실제 코트 영상 holdout set으로 공 detector 또는 sports-ball detector를 교체·benchmark해야 합니다.

## 6. canonical 3D publication gate

선수 3D model을 `verified_3d`로 발행하려면 아래 항목이 모두 충족되어야 합니다.

1. `verified_2d` provenance를 충족한 동일 선수 reference clip이 세 개 이상 있어야 합니다.
2. 동일 trial의 카메라가 intrinsic/extrinsic calibration과 synchronization 검증을 통과해야 합니다.
3. 공·손 분리·상향 궤적을 통해 슛 이벤트가 확인되어야 합니다.
4. DLT triangulation의 camera별 reprojection RMSE, bone length variation, temporal velocity spike gate를 통과해야 합니다.
5. canonical model 파일과 그 validation report가 함께 versioned 되어야 합니다.
6. 코치 또는 운동 경험 검토자가 multi-camera overlay를 보고 수동 검토를 승인해야 합니다.

단일 클립에서 예쁘게 보이는 skeleton, 자동 검색 제목, pose continuity, 또는 자체 similarity score만으로는 이 gate를 통과할 수 없습니다.

## 7. 검증 및 운영 체크리스트

개발자는 아래 명령으로 code gate를 점검합니다.

```bash
pip install -r requirements.txt
pip install -r requirements-data.txt
python -m unittest discover -s tests -v
```

운영자는 매 capture session마다 raw video, calibration file, sync evidence, reference manifest, API quality report를 같은 `capture_id` 아래에 보관합니다. 승인 source의 URL이나 라이선스 상태가 변경되면 해당 profile을 다시 `unverified_legacy`로 내리고, 기존 match 결과를 재생성해야 합니다.

## 참고문헌

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10586693/ "OpenCap: Human movement dynamics from smartphone videos"
[2]: https://www.mdpi.com/1424-8220/25/13/4003 "RTMPose-based markerless motion capture for 3x3 basketball"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/ "Pose2Sim: End-to-End Workflow for 3D Markerless Sports Kinematics"
[4]: https://www.theiamarkerless.com/industries/sports-motion-capture "Theia Markerless sports motion capture"
