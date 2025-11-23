# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**stable-diffusion-webui-eyemask** is a Python extension for AUTOMATIC1111's Stable Diffusion Web UI. It creates masks for eyes, faces, and bodies using dlib facial landmarks or MiDaS depth estimation, then redraws those masked areas using configurable prompts and parameters.

## Architecture

### Entry Points (scripts/)

Three parallel entry points provide different execution modes:

- **`em_script.py`** - Standard mode: `EyeMaskScript` extends `scripts.Script`, executes via `run()` method
- **`em_script_embedded.py`** - Embedded mode: `EyeMaskEmbeddedScript` hooks into the generation pipeline via `process()`/`postprocess()` instead of running separately. Toggled via `em_show_embedded_version` setting
- **`em_api.py`** - REST API at `/sdapi/v1/eyemask/v1/*` for programmatic access

### Core Logic (scripts/eyemask/)

- **`script.py`** - `EyeMasksCore.execute()` orchestrates the main flow: generate/receive image → generate mask → inpaint with mask prompt → save with metadata
- **`script_embedded.py`** - `EyeMasksEmbeddedCore` extends `EyeMasksCore`, applies mask redraw as a post-processing refinement step
- **`mask_generator.py`** - Mask detection algorithms:
  - `get_eyes_mask_dlib()` - Uses dlib landmarks 36-48 for eye regions
  - `get_face_mask_dlib()` - Uses dlib landmarks 0-27 for face outline
  - `get_face_mask_depth()` / `get_body_mask_depth()` - MiDaS depth estimation
- **`ui.py`** - `EyeMaskUI` renders Gradio interface, supports both standard and embedded modes
- **`wildcards.py`** - `WildcardsGenerator` handles `__wildcard_name__` pattern substitution from files in `/wildcards/`
- **`state.py`** - `SharedSettingsContext` manages model switching with automatic restoration

### Modules (scripts/eyemask/modules/)

- **`depthmap.py`** - `SimpleDepthMapGenerator` wraps MiDaS models (DPT Large, MiDaS v21/v21_small) with lazy loading and GPU/CPU selection

### API (scripts/eyemask/api/)

- **`api.py`** - FastAPI endpoints: `/mask_list`, `/generate_mask`, `/static/{path}`, `/config.json`
- **`models.py`** - Pydantic request/response models
- **`utils.py`** - Base64 image encoding/decoding

## Key Patterns

**Dual-Mode Execution**: Standard mode runs as a separate script; embedded mode hooks into the generation pipeline for seamless integration.

**Mask Types as List Indices**: Mask types stored in a list with constants for easy enable/disable.

**Context Manager for Model Switching**: `SharedSettingsContext` preserves and restores checkpoint/VAE settings when temporarily switching models.

**Wildcard System**: Files in `/wildcards/` directory; `_each` suffix enables sequential (vs random) line selection.

**Dynamic Parameter Tracking**: `EM_DYNAMIC_PARAMS` list enables regex-based metadata updates in saved images.

## Development

**Dev Mode**: Enable `em_dev_mode` in settings to show a reload button that reloads all dynamic modules without restarting the UI.

**Settings**: Registered in `em_settings.py`, persisted via web UI shared opts and JavaScript LocalStorage.

**Contributing**: Submit PRs to the `develop` branch.

## Dependencies

- **dlib** (19.24.0) - Facial landmark detection
- **cmake** - Required for dlib compilation (may need manual install)
- **MiDaS** - Cloned automatically on first use for depth estimation
- Pre-trained model `shape_predictor_68_face_landmarks.dat` downloaded to `/models/`
