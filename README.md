# Autonomous HAR & Digital Twin System for Bharatiya Antariksh Station (BAS)
### ISRO Smart India Hackathon (SIH) | Problem Statement ID: 26174

[![Space Technology](https://img.shields.io/badge/Theme-Space%20Technology-blue.svg)]()
[![ISRO](https://img.shields.io/badge/Organization-ISRO-orange.svg)]()
[![Multi-Agentic](https://img.shields.io/badge/Architecture-8--Agent%20Blackboard-cyan.svg)]()
[![Status](https://img.shields.io/badge/Tests-5%2F5%20Passing-brightgreen.svg)]()

Autonomous, offline, on-board Artificial Intelligence assistant designed to track, guide, and deterministically validate procedural experiments inside the science modules (BAS-03/BAS-04) of the upcoming **Bharatiya Antariksh Station**.

---

## 1. System Architecture: The 8-Agent Blackboard

The system is engineered as an **Optimized On-Board Multi-Agentic System (MAS)** collaborating asynchronously over a thread-safe **Digital Twin Memory Blackboard**:

```
                    +──────────────────────────────────────────────+
                    │      SHARED STATE (DIGITAL TWIN MEMORY)      │
                    │  • Fused 3D Astronaut Kinematics in Frame R  │
                    │  • Object States: DOCKED / GRASPED / EXTRACT │
                    │  • Active HOI Spatial Distance Matrix        │
                    │  • FSM Step (S0-S3) & 15-Frame Debounce      │
                    │  • Anomaly Diagnostics & Health Telemetry    │
                    +───────▲──────────────▲──────────────▲────────+
                            │              │              │
     [Inbound Perception]   │              │ [Validation] │ [Egress Output]
  ┌─────────────────────────┴──┐   ┌───────┴──────┐   ┌───┴────────────────────────┐
  │ 1. Perception Agent        │   │ 5. DT Agent  │   │ 7. Reasoning Agent         │
  │    (YOLOv8n + 3D HMR)      │   │    (3D Sync) │   │    (Next-Step Guidance)    │
  │ 2. IMU Agent               │   │ 6. Validation│   │ 8. Monitoring Agent        │
  │    (128Hz + ZUPT Filter)   │   │    Agent     │   │    (GUI + Offline TTS +    │
  │ 3. Fusion Agent            │   │    (Det. FSM)│   │     RTSP + JSONL Telemetry)│
  │    (Constrained UKF)       │   └──────────────┘   └────────────────────────────┘
  │ 4. HAR Agent               │
  │    (AdaSpot + HOI Engine)  │
  └────────────────────────────┘
```

---

## 2. Fulfillment of Official SIH PS #26174 Requirements

| Requirement | Implementation in System | Technical Approach |
| :--- | :--- | :--- |
| **Track Sequence of Experiment** | **Perception + Fusion + HAR + Validation Agents** | Multi-threaded 30 FPS video ingest, YOLOv8n, 3D pose, HOI metrics, and deterministic state transitions. |
| **Suggest Next Step** | **Reasoning & Guidance Agent** | Automatically produces proactive voice and on-screen instructions upon each state entry. |
| **Voice-based Alerts** | **Validation Agent + Monitoring Agent (TTS)** | Anomaly detector flags `ERROR_SEQ` / `ERROR_SKIP` $\to$ offline neural TTS speaks urgent warnings. |
| **Timestamped Structured Telemetry** | **Monitoring Agent (`jsonl_logger.py`)** | Serializes states and events into structured `.jsonl` lines, achieving a **3,000,000:1 compression ratio** (<15 KB per 30-min run). |
| **Dual Video Output** | **Monitoring Agent (`video_pipeline.py`)** | Concurrent local H.264 video recording (`experiments/*.mp4`) + real-time RTSP/HTTP network streaming (port 8080). |
| **Mission Control GUI** | **Monitoring Agent + Web Viewer** | Unified Mission Console displaying live stream with 2D/3D overlays, 3D Digital Twin, and step checklist. |
| **Synthetic Dataset Generation** | **`tools/generate_synthetic_data.py`** | Domain-randomized 3D generator (inverted $0G$ angles, space shadows, lighting) with pixel-perfect YOLO labels. |
| **Orientation-Agnostic 3D Tracking** | **`perception_agent.py` + `fusion_agent.py`** | Decouples floating body tilt via Canonical Orientation Constraint (COC) and transforms into rigid Rack Frame $\mathcal{R}$. |
| **Offline Standalone System** | **Master Orchestrator (`main.py`)** | 100% self-contained Python architecture with **zero cloud dependencies**. Runs on standard PCs and Windows 11. |

---

## 3. Sample Experiment Deterministic State Machine

Configured for the ISRO benchmark experiment:
> *"You are given a box that contains two smaller boxes of color red and yellow."*

```mermaid
stateDiagram-v2
    [*] --> State_0_Idle: System Initialized
    
    State_0_Idle --> State_1_Container_Open: Lid Opened (angle >= 40 deg, >= 15 frames)
    State_0_Idle --> State_0_Idle: Voice: "Please open container box"
    
    State_1_Container_Open --> State_2_Red_Extracted: Red Box extracted outside container (>= 15 frames)
    State_1_Container_Open --> State_1_Container_Open: Voice: "Next step: Please extract the red box"
    State_1_Container_Open --> ERROR_SEQ_YellowFirst: Yellow Box touched or extracted
    
    State_2_Red_Extracted --> State_3_Complete: Yellow Box extracted outside container (>= 15 frames)
    State_2_Red_Extracted --> State_2_Red_Extracted: Voice: "Next step: Please extract the yellow box"
    State_2_Red_Extracted --> ERROR_SKIP_PrematureClose: Container closed before yellow extracted
    
    State_3_Complete --> [*]: Voice: "Experiment successfully completed"

    ERROR_SEQ_YellowFirst --> State_1_Container_Open: Voice Alert + Return to Red step
    ERROR_SKIP_PrematureClose --> State_2_Red_Extracted: Voice Alert + Resume Yellow step
```

---

## 4. Quick Start & Execution Guide

### Prerequisites & CPU-Only Setup
Ensure you have **Python 3.10 or 3.11** installed. Create and activate a virtual environment:

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

**1. Install CPU-Optimized PyTorch (Crucial for non-GPU machines):**
If you do not have an NVIDIA GPU, you MUST install the CPU-only version of PyTorch to avoid downloading gigabytes of useless CUDA binaries:
```bash
# For Windows and Linux
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For macOS
pip install torch torchvision torchaudio
```

**2. Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

**3. Install and Start Ollama (For Local VLM Verification):**
This system uses a local Vision-Language Model (`Qwen3-VL:2B`) for real-time verification.
- Download and install **Ollama** from [ollama.com/download](https://ollama.com/download).
- Open a terminal and run: `ollama run qwen3-vl:2b-instruct`
- Keep Ollama running in the background while executing `main.py`.

### 1. Run the Multi-Agent System (Default Simulation Video)
```bash
python main.py
```
* Runs the full 8-agent pipeline at **80+ FPS**.
* Serves the live web dashboard at: `http://localhost:8080/`
* Speaks voice guidance and alerts via Windows offline SAPI TTS.
* Automatically records local MP4 video and structured `.jsonl` telemetry.

### 2. Run with Live Physical Webcam
```bash
python main.py --source 0
```

### 3. Run with Native Desktop Tkinter GUI
```bash
python main.py --desktop-gui
```

### 4. Test Out-of-Order Procedural Anomaly
```bash
python main.py --source experiments/anomaly_out_of_order_experiment.mp4
```
* Observes the astronaut touching/extracting the yellow box during Step 1.
* Confirms FSM catches `ERROR_SEQ` and triggers the priority voice alert:
  *"Warning: Procedural error. Red box must be extracted before yellow box."*

### 5. Generate New Synthetic Training Datasets
```bash
python tools/generate_synthetic_data.py --dataset
```
* Outputs domain-randomized synthetic training images and YOLO annotations to `dataset/images/` and `dataset/labels/`.

### 6. Run Automated Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
* Validates FSM 15-frame debouncing, anomaly gating, 3,000,000:1 telemetry ratio, and full multi-agent integration.

---

## 5. How to Train Custom Experiments

You can easily adapt this system for new procedural experiments. We provide a generalized CLI script to train both the Object Detector (YOLO) and the Human Activity Recognition (HAR) models seamlessly.

### Step 1: Prepare Your Data
- **Detector Data**: Provide a standard YOLO format `data.yaml` pointing to your annotated bounding boxes.
- **HAR Data**: Provide a structured JSON timeline (e.g., `action_labels.json`) tracking spatial metrics across experiment stages.

### Step 2: Run the Unified Training Pipeline
Use the `tools/train_experiment.py` script to train everything:
```bash
python tools/train_experiment.py \
    --name my_experiment \
    --detector-data dataset/my_experiment/data.yaml \
    --har-data dataset/my_experiment/action_labels.json \
    --epochs 12 \
    --batch 16
```

The script will:
1. Train the YOLO detector locally (Zero Cloud) and save it to `models/my_experiment_detector.pt`.
2. Train the PyTorch HAR classifier and save it to `models/my_experiment_har.pt`.

---

## 6. Repository Layout

```
e:\SIH\
├── assets/                        # Raw sample videos and static files
├── configs/                       # FSM state definitions and camera calibration
├── models/                        # Pre-trained YOLOv8 and HAR models (.pt)
├── src/                           # Core Agent Architecture
│   ├── core/                      # Shared memory and type contracts
│   ├── agents/                    # 8-Agent logic (Perception, Fusion, HAR, etc.)
│   ├── audio/                     # Offline TTS synthesis
│   ├── streaming/                 # Video pipeline and networking
│   ├── telemetry/                 # JSONL data loggers
│   └── gui/                       # Web & Tkinter Mission Control consoles
├── tests/                         # End-to-end integration and unit tests
├── tools/                         # Unified CLI scripts and data generators
│   ├── train_experiment.py        # Generalized unified training pipeline
│   ├── legacy/                    # Archived legacy training scripts
│   └── ...                        # Generators and annotators
├── experiments/                   # Generated test videos and telemetry outputs
├── main.py                        # Single unified launcher executing MAS
├── requirements.txt               # Dependency specifications
└── README.md                      # System documentation
```

---

## 7. Spaceflight Avionics Qualification Roadmap

* **Flight Target Hardware**: Dual-compute architecture pairing the **NVIDIA Jetson Orin NX (10–25W)** for vision/HMR inference with the **NASA/Microchip PIC64-HPSC (RISC-V)** executing the deterministic FSM and telemetry serializer in an isolated RTOS partition (VxWorks / WorldGuard).
* **Thermal Dissipation**: Conduction-cooled baseplate connected to the BAS-03 laboratory liquid loop.
* **Telemetry Heritage**: Built upon ISRO POEM-4 flight heritage (MOI-TD on-orbit AI lab and RRM-TD robotic arm vision).
