# FREEZE  
> 방향 인지·환경음 인지 기반 청각장애인 안전 보조 시스템  
> Direction-aware Audio-Visual Safety Assistant

## 🔍 프로젝트 개요 (Overview)

**FREEZE**는 청각 정보에 취약한 사용자의 안전을 보조하기 위해 만든 시스템입니다.  

- **환경음 분류 (YAMNet)**  
  - 사이렌, 경보/알람(차량 경적, 화재 경보기 등)을 실시간 분류  
  - 위험도에 따라 이벤트 발생

- **음성 인식 (Whisper)**  
  - 음성 구간만 자동 탐지(VAD + 게이트)  
  - 필요 시 발화 내용을 텍스트로 변환하여 로그/앱에 전송

- **카메라 기반 객체 감지 (YOLO)**  
  - 사람/차량 등 특정 객체 감지 시 스냅샷 및 짧은 클립 자동 저장  
  - 위험 이벤트와 연동해 영상 증거 확보

- **방향 인지 & 진동 피드백**  
  - ReSpeaker Mic Array DoA(Direction of Arrival)를 이용해 소리 방향(0–359°) 추정  
  - 서버 → ESP32 쪽으로 방향·위험 정보를 내려 보내,  
    사용자의 몸에 부착된 여러 개의 진동 모터를 **각도별로 다르게 진동**시키는 구조

- **앱/Web 클라이언트**  
  - `/ws/app` WebSocket으로 실시간 이벤트(스냅샷, 클립 URL, 환경음 분류, 음성 인식 결과 등) 수신  
  - UI는 별도 프로젝트로 분리 가능

---

## 🧩 시스템 구성 (Architecture)

간단한 데이터 플로우는 다음과 같습니다.

1. **라즈베리파이 + ReSpeaker + 카메라**
   - 오디오: 16kHz mono PCM int16, 200ms 프레임(6400 bytes)로 스트리밍  
   - 비디오: JPEG 프레임(기본 640×640, 15fps) 전송  
   - IR 리모컨 입력 → 특정 키워드를 서버로 전송 (`type: "ir"`)  
   - UART로 ESP32와 통신(서버 접속이 안 될 때 로컬 fallback)

2. **AWS EC2 상 FastAPI 서버 (이 저장소)**  
   - `routes_ws.py`
     - `/ws/audio` : YAMNet + Whisper + 방향/위험 이벤트 처리
     - `/ws/camera` : YOLO 기반 객체 감지 + 스냅샷/영상 녹화
     - `/ws/esp32` : ESP32와 WebSocket 연동(진동 제어 등)
     - `/ws/app`   : 앱/Web 클라이언트 구독용 허브(topic 기반)
   - `routes_http.py`
     - 스냅샷/영상 업로드, 헬스체크 등 HTTP API
   - `runtime.py`
     - 전역 상태 관리(마지막 방향, 분류 결과, dBFS, transcript 등)
     - `broadcast_info()` : 모든 이벤트를 한 곳에서 포맷팅해서 앱/ESP32로 브로드캐스트

3. **ESP32 + 진동 모듈**
   - 서버에서 받은 `direction`, `group_label`, `group_conf`, `ms`를 기반으로  
     각 모터 핀을 매핑해서 **각도별 진동 패턴**을 만들어냄.

---

## 📁 폴더 구조 (대략)

```text
FREEZE/
 ├─ app.py                # FastAPI 엔트리 포인트
 ├─ runtime.py            # 전역 상태, 공용 유틸, broadcast_info 등
 ├─ camera_module.py      # YOLO 기반 카메라 이벤트 처리
 ├─ danger_check.py       # 환경음 그룹별 위험도 판단 로직
 ├─ whisper_module.py     # Whisper STT 래퍼
 ├─ yamnet_module.py      # YAMNet 환경음 분류 래퍼
 ├─ yolov8n.pt            # YOLO 모델(환경변수로 경로 변경 가능)
 ├─ routes/
 │   ├─ __init__.py
 │   ├─ routes_ws.py      # WebSocket 엔드포인트 (/ws/audio, /ws/camera, /ws/esp32, /ws/app)
 │   └─ routes_http.py    # HTTP 엔드포인트 (파일 업로드 등)
 ├─ clips/                # 스냅샷 · 녹화 영상 · 오디오 샘플 저장 폴더
 ├─ uploads/              # (필요 시) 업로드 파일 폴더
 ├─ labels/
 │   └─ class_labels_indices.csv  # YAMNet 라벨 매핑
 ├─ .env                  # 환경변수 설정
 ├─ requirements.txt
 └─ README.md
