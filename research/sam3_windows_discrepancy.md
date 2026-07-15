# SAM 3 Windows Import Discrepancy Report

During the validation of the official Meta `sam3` package imports, a critical dependency conflict was discovered that prevents loading the package on Windows systems.

---

## 1. Description of Discrepancy

While the repository's `pyproject.toml` lists standard cross-platform libraries (such as `timm`, `numpy`, `huggingface_hub`), the core model implementation has a hard module-level dependency on **Triton** (via `sam3/sam3/model/edt.py`) that is triggered immediately upon importing any model builder function.

### Traceback of the Import Failure
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\tarun\OneDrive\Desktop\adfusionR&D\sam3\sam3\model_builder.py", line 40, in <module>
    from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
  File "C:\Users\tarun\OneDrive\Desktop\adfusionR&D\sam3\sam3\model\sam1_task_predictor.py", line 16, in <module>
    from sam3.model.sam3_tracker_base import Sam3TrackerBase
  File "C:\Users\tarun\OneDrive\Desktop\adfusionR&D\sam3\sam3\model\sam3_tracker_base.py", line 10, in <module>
    from sam3.model.sam3_tracker_utils import get_1d_sine_pe, select_closest_cond_frames
  File "C:\Users\tarun\OneDrive\Desktop\adfusionR&D\sam3\sam3\model\sam3_tracker_utils.py", line 9, in <module>
    from sam3.model.edt import edt_triton
  File "C:\Users\tarun\OneDrive\Desktop\adfusionR&D\sam3\sam3\model\edt.py", line 8, in <module>
    import triton
ModuleNotFoundError: No module named 'triton'
```

---

## 2. Technical Analysis

1.  **Triton Platform Restrictions**: Triton is officially supported only on Linux. There are no official pre-built wheels or native installation channels for Triton on Windows.
2.  **Unconditional Module-Level Imports**: The file `sam3/sam3/model/edt.py` imports `triton` and `triton.language as tl` unconditionally at the top of the file:
    ```python
    import torch
    import triton
    import triton.language as tl
    ```
    Because this import occurs at the module level rather than inside the execution function, Python attempts to load it immediately during the module initialization phase.
3.  **Import Path Dependency**: Any attempt to load the main model entry points (such as `build_sam3_image_model` or `build_sam3_predictor`) imports the `SAM3InteractiveImagePredictor`, which transitively imports `edt.py` through the tracker utilities, causing the entire package to crash on Windows.

---

## 3. Options for Resolution

Since the instructions explicitly forbid creating compatibility wrappers or modifying downstream interfaces without approval, we present the following options to proceed:

### Option A: Patch the Upstream File `sam3/sam3/model/edt.py`
We can edit the official repository code locally to handle the `triton` import gracefully with a fallback (since the Euclidean Distance Transform has a CPU fallback in OpenCV/scipy anyway):
```python
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
```
*Note: This modifies the upstream code of the cloned repository, which violates "Build against the upstream implementation as directly as possible" unless explicitly approved.*

### Option B: Install an Unofficial Triton Windows Fork
We can attempt to install an unofficial community build of Triton for Windows (e.g. `pip install triton-windows`).
*Warning: Community builds of Triton on Windows are highly unstable, compile-heavy, and frequently fail due to CUDA Toolkit and MSVC C++ compiler mismatches on Windows 11 host systems.*

### Option C: Migrate the Pipeline Execution to WSL (Windows Subsystem for Linux)
Run the python pipeline inside a WSL environment where CUDA and Triton are natively supported on Linux.
