# System Report: Live Procedure Validation + RHINO Human–Object Reconstruction on DGX Spark

**Date:** 2026-11-07  
**Branch:** `dgx-spark` (all changes uncommitted)  
**Host:** NVIDIA DGX Spark: aarch64, NVIDIA GB10 GPU (compute capability 12.1 / sm_121), 20 CPU cores, 121 GB unified memory, CUDA 13.0, no local camera.

---

## 1. Summary

| Area | Status | Key result |
|---|---|---|
| Live pipeline with a laptop camera streamed through the browser | **Working** | 15–24 FPS end-to-end on CUDA from a remote browser webcam |
| Water-bottle experiment (6-state FSM) in the live pipeline | **Integrated, not yet validated by a full human run** | Detector auto-switches; all 3 experiments switchable without restart |
| Detection of every object in view, with who is holding what | **Working** | YOLOE prompt-free, 4,585 classes, ~9 ms/frame; 25–31 FPS with full pipeline |
| React dashboard (camera, experiment switcher, scene objects) | **Working** | Type-checks clean (`tsc -b`); camera capture no longer freezes |
| RHINO offline 3D reconstruction: core install | **Working** | GPU geometry smoke test exact; pretrained demo re-render matches reference |
| RHINO preprocessing (own videos) | **5 of 6 tool environments ready** | Blocked on 3 licensed files + 1 user decision |

---

## 2. Live Analysis System

### 2.1 Architecture

```
Laptop browser (getUserMedia, 1280×720)
   │  JPEG q=0.8, one in-flight POST at a time
   ▼
:8081  BrowserCapture  (cv2.VideoCapture-compatible, newest-frame-wins)
   ▼
main.py orchestrator (CUDA)
   ├─ PerceptionAgent      pose + protocol detector (bottle model / generic model, hot-swappable)
   ├─ SceneObjectAgent     YOLOE-11s-seg prompt-free → all objects + hand interaction
   ├─ ValidationAgent      per-experiment finite-state machine (JSON config)
   └─ MonitoringAgent      telemetry (incl. scene_objects)
   ▼
:8080  MJPEG /stream, /telemetry, /api/experiment, /api/source, /reset
   ▼
:5173  React dashboard (Vite)  ← also hosts the camera capture
```

### 2.2 Measured performance

| Configuration | Throughput |
|---|---|
| Browser webcam → full pipeline (bottle config, CUDA) | **15–24 FPS** (bounded by upload rate over the remote link) |
| Recorded clip (`clip1.mp4`) → full pipeline incl. scene objects | **25–31 FPS** |
| YOLOE scene-object inference alone | **≈ 9 ms / frame** (re-detect every 2nd frame; interactions every frame) |

### 2.3 Water-bottle experiment (`configs/water_bottle_fsm.json`)

States: `IDLE → BOTTLE_TAKEN → CAP_OPENED → DRINKING → CAP_CLOSED → COMPLETE`

| Parameter | Value |
|---|---|
| Debounce | 6 frames |
| Mouth-contact threshold | 0.12 m |
| Bottle tilt threshold | 22° |
| Hand-contact threshold | 0.10 m |
| Per-step timeout | 120 s |
| Forbidden events enforced | e.g. drinking before cap opened; closing before drinking |

**Validation status:** the pipeline, detector swap, dashboard protocol and FSM wiring were verified. **A full live run by a human operator has not been recorded**: no completed-run telemetry exists. At the time of writing the backend is running but idle (no frames being streamed).

### 2.4 Scene objects and interactions

- Model: `models/yoloe-11s-seg-pf.pt`, 4,585 open-vocabulary classes, confidence ≥ 0.30, at most 25 objects.
- The model filters out people and body parts, room surfaces, and boxes covering more than 60% of the frame.
- Interaction label, from the distance between a wrist keypoint or the `operator_hand` box and an object's box:
  - **HOLDING**: the hand point is inside the box.
  - **TOUCHING**: distance < max(25 px, 0.15 × box diagonal).
  - **NEAR**: distance < max(80 px, 0.5 × box diagonal).
  - **NONE**: otherwise.
- The strongest interaction becomes the activity text (e.g. `HOLDING CUP`) when the FSM is idle. Scene objects never drive FSM transitions.

### 2.5 Defects found and fixed

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | 8080 dashboard blank white page | Corrupted `<title>` line (commit f25afd7) | Restored head markup |
| 2 | Bottle experiment absent on 8080 | No bottle case in `/api/experiment` or the UI | Added endpoint case, button and 6-step protocol |
| 3 | Clothing labelled `red_box` during the bottle run | Red/yellow colour fallback ran with the bottle model | Fallback disabled when the bottle model is active |
| 4 | Backend hang on experiment switch | Re-creating PerceptionAgent deadlocked in `RelateAnything.from_pretrained` | `load_detector()` hot-swaps only the YOLO weights |
| 5 | Camera freezes in the dashboard | Browser throttles background tabs; capture lived in a separate tab | Capture moved into the React tab (`useBrowserCamera`); CORS on :8081 |
| 6 | No way to switch experiments in React | Missing | Header experiment switcher driven by telemetry `experiment_id` |

---

## 3. RHINO Offline Reconstruction

RHINO (CVPR 2026) optimises one video at a time into a 4D reconstruction of a human, an object and the scene. It does **not** run in real time. It complements the live system as an offline analysis stage.

### 3.1 Core install (rhino env, Python 3.11, torch 2.14.1+cu130)

| Component | Version | Note |
|---|---|---|
| PyTorch3D | 0.7.9 | Source build; `setup.py` C++17 → C++20 (required by torch 2.14 headers) |
| Kaolin | 0.18.0 | Source build; patched `mise.cpp` `at::zeros` call for the new ATen API |
| RHINO | editable | |
| pycolmap | 3.12.6 (conda-forge) | README pins 3.10.0, which has no aarch64 wheel |

**GPU geometry smoke test** (unit cube; query points at the centre and at x = 1.5):

| Check | Expected | Measured |
|---|---|---|
| Squared point-to-mesh distance | 0.25, 1.0 | **0.25, 1.0** |
| Inside test (`check_sign`) | True, False | **True, False** |

### 3.2 Pretrained demo re-render (`demo_table`, checkpoint `0209-098070.ckpt`)

| Metric | Value |
|---|---|
| Frames rendered (test subset) | 5 of 467 |
| Wall time | 7 min 19 s (**≈ 88 s / frame**, GPU shared with the live backend) |
| Projected full sequence | ≈ 11 h |
| Mean absolute pixel difference vs the authors' renders (0–255 scale) | **0.01** on all 5 frames |
| Outputs per frame | RGB / normal / depth comparisons; human, object and background renders; deformed human mesh; object and scene `.ply` |

Output: `third_party/rhino_demo_rerender/demo_table/hoi/pretrained/epoch_0000/`

**Conclusion:** the reconstruction on DGX Spark reproduces the reference results to within rounding.

### 3.3 Fixes required on this platform

| # | Failure | Fix |
|---|---|---|
| 1 | Checkpoint load: size mismatch in `smplx_body/feet_interior_points` (3333/563 vs 3340/556) | Body/feet split is a y = −1.05 cut; 7 boundary vertices flip because of floating-point differences. `HOINetwork._load_from_state_dict` now adopts the checkpoint's buffers. |
| 2 | `'numpy.ndarray' object has no attribute 'ptp'` | numpy 2 removed `.ptp()`, and RHINO pins trimesh < 4. All 23 trimesh call sites rewritten to `np.ptp()`; original backed up in `third_party/trimesh_backup_pre_ptp/`. |
| 3 | A full render takes ~11 h | Added `+limit_test_batches=N` option to `train.py` |

### 3.4 Preprocessing toolchain (for the user's own videos)

Every environment uses torch 2.14.1 + cu130 instead of the README's torch 2.1 / 2.4 + cu121, which have no builds for aarch64 / sm_121.

| Step / tool | Env | Status | Verification |
|---|---|---|---|
| Masks: SAM2 + GroundingDINO | `sam2` (py3.11) | Ready | GPU OK; `sam2_hiera_large.pt` (898 MB) downloaded |
| Camera/object SfM: COLMAP 3.12.6, GLOMAP 1.2.0 | `colmap` (conda-forge) | Ready | Binaries run (CUDA build) |
| Features/matching: hloc 1.5, LightGlue | `rhino` | Ready | Preprocessing modules import |
| MASt3R matching | `mast3r` (py3.11) | Ready | curope CUDA kernel output equals the reference (max error 0.0) |
| Depth/normals: Metric3D ViT-S | `rhino` | Ready | Depth and normal maps produced on GPU |
| SMPL-X fit: AiOS + mmcv-full 1.6.1 | `aios` (py3.10, numpy 1.23.5) | Ready | mmcv CUDA NMS OK; deformable-attention forward matches PyTorch (float max abs error 4.7e-10); checkpoint SHA-256 verified |
| Keypoints: Sapiens | `sapiens` (py3.10) | Env ready; **pose weights missing** | mmcv/mmpose/mmdet import with CUDA; RTMDet detector downloaded and SHA-256 verified |
| Contact: InteractVLM | — | **Not started** | Needs registration |

**Known limitation:** AiOS's deformable-attention *backward* kernel exceeds launch resources on sm_121 in the large-channel test. Preprocessing runs inference only, which uses the verified forward kernel.

### 3.5 Blocking items (user action)

1. **`SMPLX_to_J14.pkl`:** in `expose_data.zip` (SMPL-X site, ExPose downloads).
2. **`SMPL_NEUTRAL.pkl`:** smpl.is.tue.mpg.de; place in `third_party/RHINO/third_party/AiOS/data/body_models/smpl/`.
3. **InteractVLM data:** register at interactvlm.is.tue.mpg.de and run `fetch_data.sh` yourself (it asks for credentials).
4. **Sapiens COCO whole-body 2B weights (8.3 GB):** only on a third-party mirror (`noahcao/sapiens-pose-coco`), with no official hash. Needs a decision before download.

---

## 4. Risks and Open Issues

| Risk | Impact | Mitigation |
|---|---|---|
| Live bottle FSM not validated on a real run | Unknown step-detection accuracy | Record 3–5 operator runs; review telemetry per state |
| RHINO ≈ 88 s/frame | Not usable live; minutes per second of video | Run offline on short clips; stop the live backend while rendering |
| Local patches to third-party code (Kaolin, PyTorch3D, mmcv, Sapiens cv, AiOS ops, trimesh, RHINO) | Lost on reinstall | Record them as patch files in the repo |
| Extra torch install left in miniforge base env | Clutter, possible import confusion | Uninstall if unused |
| Unexplained "switch to Red-Yellow" request seen once during a live session | Unexpected experiment change | Monitor `/api/experiment` requests |

---

## 5. Next Steps

1. Do a live water-bottle run on the React dashboard (`localhost:5173` → *Start Camera* → *Water Bottle*), then analyse the per-state telemetry.
2. Supply the four blocking items in §3.5, then build the InteractVLM env.
3. Run full RHINO preprocessing and training on a 10–20 s bottle clip recorded with a slowly moving camera.
4. Commit the live-system changes and save the third-party patches as files.
