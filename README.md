# Tesla FSD Occupancy Stream — CUDA Voxel Grid & Go BEV Transform 🚗

> **CUDA 3D voxel grid raycasting kernel and Go Bird's-Eye-View (BEV) coordinate transformer for Tesla FSD.**

[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76B900)]()
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![Domain](https://img.shields.io/badge/Domain-Autonomous%20Driving-red)]()

---

## 🎯 For Recruiters & Hiring Managers

This repository implements the **Tesla FSD Occupancy Stream** — processing 8-camera surround vision streams into 3D occupancy grids and Bird's-Eye-View (BEV) coordinates. It demonstrates:

- **CUDA 3D voxel raycasting kernel** building 200x200x16 3D occupancy grids with 23 semantic classes
- **Go BEV coordinate transformer** converting world-space coordinates to ego-centric grid cells with temporal fusion
- **Multi-camera feature projection** mapping 2D pixel features to 3D world positions in real time
- **Python simulation test harness** verifying occupancy grid predictions

**Why this matters**: Vision-based autonomous driving requires converting raw multi-camera video into real-time 3D occupancy representations for path planning and obstacle avoidance.

---

## 🔬 For Engineers & Technical Reviewers

### Core Components

| Component | Language | Purpose |
|---|---|---|
| `src/occupancy_grid.cu` | CUDA | CUDA kernel for voxel raycasting & semantic logit accumulation |
| `src/bev_transform.go` | Go | Ego-relative world↔grid coordinate transformer |
| `tests/` | Python | Occupancy grid reconstruction test harness |

---

## 🤖 ML/AI & Programmatic Mesh Integration

- **MCP Tool**: `query_occupancy_grid()` — 3D occupancy status queryable by driving agents
- **Mastermind Sidecar**: Fully connected to APEX Highway mesh
- **SHA-256 Integrity**: Tracked in `.integrity/file_hashes.json`

---

## ⚡ Quick Start

```bash
python3 tests/test_occupancy.py
```
