# ArmyBot — ROS 2 + Vision + Voice Robotics Automation

> ROS 2, AI Vision, Depth Camera, Arduino, VLM, 협동로봇을 통합한 군 사격 훈련 자동화 프로젝트

## Overview

ArmyBot은 사격 훈련 과정에서 발생하는 **표적 분석의 수작업 의존성**, **탄피·탄알집 수거 인력 소모**, **훈련 결과의 비정량적 관리**를 줄이기 위해 개발한 자동화 시스템입니다.

Doosan M0609 협동로봇을 중심으로 음성 명령, YOLO 객체 탐지, RealSense Depth, Arduino 이벤트 감지, VLM 기반 표적 분석, Flask UI를 연결해 다음 흐름을 구현했습니다.

```text
Voice Command
    ↓
Robot Ready Pose
    ↓
Arduino Shot Event
    ↓
YOLO + RealSense Detection
    ↓
Depth → Robot Coordinate Transform
    ↓
Robot Collection Task
    ↓
Target Capture → VLM Analysis
    ↓
Result Storage / UI
```

---

## System Components

| Component | Role |
|---|---|
| **ROS 2 Humble** | 노드 간 통신 및 전체 시스템 오케스트레이션 |
| **Doosan M0609** | 탄피·탄알집 수거 및 물리 작업 수행 |
| **YOLOv8** | 탄피·탄알집 객체 탐지 |
| **RealSense D435i** | RGB-D 데이터 및 대상 위치 추정 |
| **Arduino** | 격발 이벤트 감지 및 시리얼 통신 |
| **Gemini VLM** | 촬영된 표적 이미지 분석 |
| **Flask** | 지휘자/사수용 UI |
| **SpeechRecognition / gTTS** | 음성 명령 및 안내 |

---

## Core Features

### 1. Vision + Depth 기반 대상 위치 추정

YOLO로 탄피·탄알집을 탐지하고 RealSense Depth를 이용해 카메라 좌표계의 위치를 얻은 뒤, 캘리브레이션 행렬을 적용해 **Robot 좌표계로 변환**했습니다.

### 2. Robot Manipulation

변환된 위치 정보를 ROS 2 로봇 제어 노드로 전달해 Doosan M0609이 실제 수거 동작을 수행하도록 구성했습니다.

### 3. Arduino–ROS 2 Integration

격발 이벤트를 Arduino에서 감지하고 PySerial 기반 통신으로 ROS 2 시스템에 전달해, 이벤트 발생 이후의 자동화 플로우를 트리거했습니다.

### 4. Voice-driven Workflow

음성 명령을 시스템 상태 전환의 입력으로 사용해 사람이 복잡한 UI를 조작하지 않아도 준비·실행 흐름을 시작할 수 있도록 구성했습니다.

### 5. VLM-based Target Analysis

사격 종료 후 표적 이미지를 촬영하고 VLM으로 분석해 결과를 저장하는 흐름을 구현했습니다.

---

## My Contribution

- ROS 2 기반 로봇 제어 노드 설계 및 구현
- RealSense Depth → Robot 좌표계 변환 알고리즘 구현
- YOLO 기반 객체 탐지 파이프라인 구축
- Arduino–ROS 2 시리얼 통신 프로토콜 설계
- 음성 명령 기반 자동 실행 로직 설계
- Vision–Robot–Embedded–VLM 전체 시스템 통합 및 디버깅

---

## Tech Stack

`Ubuntu 22.04` `ROS 2 Humble` `Python` `YOLOv8` `OpenCV` `RealSense D435i` `Arduino` `PySerial` `Gemini VLM` `Flask` `Doosan M0609`

---

## Project Structure

```text
armybot/
├── robot_control.py
├── yolo_node.py
├── ai_count.py
└── onrobot.py

arduino_bridge/
└── switch_edge_pub.py

jarvis_project/
└── jarvis.py

resource/
├── brass_magazine.pt
├── calibration_matrix.yaml
└── result/

armbot_web/
├── commander.py
├── shooter.py
└── templates/
```

---

## Key Learning

이 프로젝트에서 가장 중요한 경험은 AI 모델 하나를 만드는 것이 아니라 **Vision, Depth, Robot Control, Embedded Event, Voice, VLM을 하나의 동작 가능한 시스템으로 연결하는 것**이었습니다.

각 모듈이 개별적으로 동작하더라도 좌표계, 메시지 규격, 이벤트 순서, 예외 처리 방식이 맞지 않으면 전체 서비스가 실패한다는 점을 경험하며 시스템 통합의 중요성을 배웠습니다.

---

## Team

ROKEY D-3  
이강인 · 주진 · 최순일 · 최재형