# Tesla-Style Occupancy Stream — Deterministic Simulation Study

Independent GlacierEQ portfolio work modeling occupancy-grid workload arithmetic and temporal stream integrity for autonomous-driving-style scenarios.

**Evidence state:** `MODELED_OCCUPANCY_SYSTEM_NOT_TESLA_FSD_MEASUREMENT_OR_AUTHORITY`

This repository is not affiliated with, endorsed by, or operated by Tesla. It does not access Tesla vehicles, FSD software, camera feeds, HW3/HW4 hardware, or proprietary data.

## Current mechanisms

### Occupancy workload model

`src/tesla_fsd_occupancy_stream.py` computes deterministic scenario values from explicit inputs:

- camera count;
- voxel-grid dimensions;
- modeled voxels-per-millisecond assumption;
- modeled per-camera overhead;
- target FPS;
- deadline.

The result reports modeled workload size, modeled latency, a modeled FPS upper bound, and whether the arithmetic satisfies the supplied deadline.

Every result carries:

`MODELED_OCCUPANCY_SCENARIO_NOT_TESLA_FSD_MEASUREMENT`

The implementation does not use wall-clock timing and does not label arithmetic as HW4 performance.

### Temporal occupancy integrity

`src/occupancy_integrity.py` evaluates a caller-supplied sequence of modeled occupancy frames.

It fails closed on duplicate frame identities and non-monotonic timestamps, and surfaces review state when:

- camera-view count drops below the declared quorum; or
- occupancy fraction changes more than the declared temporal threshold.

The result preserves issue codes and emits a deterministic SHA-256 receipt.

`MODELED_OCCUPANCY_INTEGRITY_NOT_TESLA_DRIVING_AUTHORITY`

No path planning, steering, braking, perception inference, or vehicle command occurs.

## Proof surfaces

| Surface | Purpose |
|---|---|
| `src/tesla_fsd_occupancy_stream.py` | deterministic occupancy workload/deadline scenario |
| `src/occupancy_integrity.py` | temporal sequence and view-quorum integrity |
| `tests/test_fsd.py` | deterministic scenario, deadline, input refusal |
| `tests/test_occupancy_integrity.py` | view loss, discontinuity, identity/time refusal |
| `.github/workflows/tests.yml` | repository-native unittest workflow |

## Native proof

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Evidence boundary

Current source does **not** establish CUDA execution, Tesla FSD integration, HW3/HW4 performance, camera perception, multi-camera projection, driving safety, vehicle actuation, production deployment, or Tesla affiliation.

The transferable engineering value is deterministic workload modeling plus explicit temporal integrity/refusal semantics around a simulation-bound occupancy stream.
