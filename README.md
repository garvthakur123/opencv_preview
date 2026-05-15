# opencv_preview

A collection of C++/CMake subprojects demonstrating GPU-accelerated image processing using OpenCV, OpenCL, and CUDA. Each subproject is self-contained with its own `CMakeLists.txt` and builds independently.

---

## Repository structure

```
opencv_preview/
├── opencv_0/               # Minimal OpenCV image viewer (no GPU)
├── parallel_worlds_1/      # OpenCL: grayscale filter on live camera
├── parallel_worlds_2/      # OpenCL: grayscale + Sobel edge detection
├── parallel_worlds_3/      # OpenCL: grayscale + Sobel + effect + dehazing  ← main demo
└── parallel_worlds_2_cuda/ # CUDA: grayscale + Sobel (Linux/Windows only)
```

---

## Subprojects

### `opencv_0` — Basic OpenCV demo
Loads `preview.png`, displays it in a window, waits for a keypress, saves it as `preview_new.png`.
- **Depends on:** OpenCV only
- **Executable:** `opencv_preview`

### `parallel_worlds_1` — OpenCL grayscale
Opens webcam (falls back to `preview.png`). Runs a grayscale kernel on the GPU and shows two windows: original and grayscale.
- **Depends on:** OpenCV, OpenCL
- **Executable:** `parallel_worlds_1`

### `parallel_worlds_2` — OpenCL grayscale + Sobel
Chains grayscale → Sobel edge detection. Two windows.
- **Depends on:** OpenCV, OpenCL
- **Executable:** `parallel_worlds_2`

### `parallel_worlds_3` — OpenCL full pipeline + dehazing ← main demo
The most complete subproject. Five simultaneous windows, all processed on the GPU via OpenCL each frame. See below for full details.
- **Depends on:** OpenCV, OpenCL
- **Executable:** `parallel_worlds_3`

### `parallel_worlds_2_cuda` — CUDA grayscale + Sobel
CUDA equivalent of `parallel_worlds_2`. Always reads from `preview.png` (camera path is disabled in code). Targets SM 75 (Turing-class GPU).
- **Depends on:** OpenCV, CUDA toolkit
- **Note:** macOS is not supported — Apple dropped CUDA support. Linux/Windows only.
- **Executable:** `parallel_worlds`

---

## parallel_worlds_3 in detail

### What it shows (5 windows)

| Window | What it shows |
|--------|--------------|
| `preview` | Raw input — live camera or still image |
| `converted` | Grayscale using ITU-R BT.709 luminance weights (`grey` kernel) |
| `edge` | Sobel edge detection on the grayscale image, with tiled local-memory NDRange (`sobel` kernel) |
| `effect` | Original colour darkened at edge locations (`effectFilter` kernel) |
| `dehazed` | **Dark Channel Prior dehazing** — haze/fog removal (`darkChannel` + `estimateTransmission` + `recoverRadiance` kernels) |

### Dehazing — what was added

The dehazing is an integration of the Dark Channel Prior algorithm (He et al., CVPR 2009) into the existing OpenCL filter pipeline. It was not part of the original project.

**New files:**

| File | Role |
|------|------|
| `src/DehazeFilter.hpp` | C++ filter class, follows the same `ImageFilter` subclass pattern as `SobelFilter` and `EffectFilter` |
| `src/dehazeFilters.cl` | Three OpenCL kernels: `darkChannel`, `estimateTransmission`, `recoverRadiance` |

**Modified files:**

| File | What changed |
|------|-------------|
| `src/main.cpp` | Added `DehazeFilter` instance, `dehazeFrame` buffer, `"dehazed"` window, `--image` CLI flag |
| `src/CMakeLists.txt` | Removed `-msse2` (x86-only flag, breaks compilation on Apple Silicon ARM64) |
| `src/OpenCLInterface.hpp` | Added `CL_HPP_TARGET_OPENCL_VERSION 120` — required for macOS, which ships OpenCL 1.2; the bundled `cl2.hpp` header defaults to 2.0 and fails to compile without this |
| `src/greyImageFilters.cl` | Restored from ROT13 obfuscation to valid OpenCL source |

**How the dehazing pipeline works:**

1. `darkChannel` kernel (GPU) — finds the minimum BGR value in a 15×15 neighbourhood per pixel
2. Atmospheric light estimation (CPU) — reads dark channel back, takes the mean of the top 0.1% brightest pixels as the haze colour `A`, capped per-channel at 0.85
3. `estimateTransmission` kernel (GPU) — computes how much haze is present at each pixel
4. `recoverRadiance` kernel (GPU) — subtracts the haze and recovers the scene

---

## Dependencies

| Dependency | Used by | Notes |
|------------|---------|-------|
| CMake ≥ 3.12 | all | |
| OpenCV 4.x | all | needs `highgui`, `imgproc` |
| OpenCL 1.2+ | `parallel_worlds_1/2/3` | see platform notes below |
| CUDA toolkit | `parallel_worlds_2_cuda` | Linux/Windows only |

---

## Building and running

Every subproject builds independently from its own `src/` directory. The pattern is the same for all of them:

```bash
cd <subproject>/src
cmake -S . -B ../build
cmake --build ../build
cd ../build
./<executable-name>
```

The build automatically copies all `.cl` kernel files and `preview.png` next to the executable. **Always run the executable from the `build/` directory**, not from `src/`.

---

### macOS — tested on Apple Silicon (M-series), macOS 14

OpenCL 1.2 ships with macOS via `OpenCL.framework` — no extra install needed. The CUDA warning during CMake configure is harmless (CUDA is not required).

**Install dependencies:**

```bash
xcode-select --install   # Xcode Command Line Tools — provides clang and OpenCL headers
brew install opencv
```

**Build and run `parallel_worlds_3`:**

```bash
cd parallel_worlds_3/src
cmake -S . -B ../build
cmake --build ../build
cd ../build
./parallel_worlds_3                          # live camera
./parallel_worlds_3 --image /path/to/img.jpg # still image
```

Press any key to quit.

> `parallel_worlds_2_cuda` does **not** build on macOS — Apple removed CUDA support. All other subprojects build and run on macOS.

---

### Linux — x86_64 (not tested; instructions based on code)

You need an OpenCL ICD loader and a vendor-specific OpenCL implementation.

**Install dependencies:**

```bash
# Ubuntu / Debian
sudo apt install cmake build-essential libopencv-dev ocl-icd-opencl-dev opencl-headers
```

For GPU-specific OpenCL:
- **NVIDIA:** install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) — includes OpenCL
- **AMD:** install ROCm or the AMDGPU-PRO driver
- **Intel integrated graphics:** `sudo apt install intel-opencl-icd`

If CMake cannot locate OpenCL automatically (no CUDA toolkit installed), pass the path manually:

```bash
cmake -S . -B ../build -DOpenCL_ROOT=/usr
```

**Build and run `parallel_worlds_3`:**

```bash
cd parallel_worlds_3/src
cmake -S . -B ../build
cmake --build ../build
cd ../build
./parallel_worlds_3                          # live camera
./parallel_worlds_3 --image /path/to/img.jpg # still image
```

**Build and run `parallel_worlds_2_cuda`** (NVIDIA GPU required):

```bash
cd parallel_worlds_2_cuda/src
cmake -S . -B ../build
cmake --build ../build
cd ../build
./parallel_worlds                            # reads preview.png from build dir
```

> If your GPU is not Turing (SM 75), edit `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` to match your GPU generation before building.

---

### Windows — not tested; instructions based on code

The `CMakeLists.txt` has a separate MSVC flag block (`/arch:SSE2`) so the ARM64 fix does not affect Windows.

**Install dependencies:**

- [CMake](https://cmake.org/download/) ≥ 3.12
- Visual Studio 2019 or 2022 with the "Desktop development with C++" workload
- [OpenCV Windows release](https://opencv.org/releases/) — note the path to the directory containing `OpenCVConfig.cmake`
- OpenCL via your GPU driver:
  - **NVIDIA:** install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
  - **AMD/Intel:** install the respective GPU driver, which includes an OpenCL ICD

**Build `parallel_worlds_3` (Developer Command Prompt):**

```bat
cd parallel_worlds_3\src
cmake -S . -B ..\build -DOpenCV_DIR="C:\path\to\opencv\build"
cmake --build ..\build --config Release
```

**Run:**

```bat
cd ..\build\Release
parallel_worlds_3.exe                           # live camera
parallel_worlds_3.exe --image C:\path\to\img.jpg  # still image
```

> The executable is placed in `build\Release\` or `build\Debug\` depending on the config. The post-build step copies `src\` and `preview.png` alongside it automatically.

---

## Input behaviour (`parallel_worlds_3`)

When launched without `--image`:
1. Tries to open webcam device `0`
2. If no camera found, reads `preview.png` from the working directory
3. If neither available, exits with code 3

The `--image` flag accepts any path and any image format supported by OpenCV (JPEG, PNG, BMP, etc.). When `--image` is used, the camera is not opened.

---

## Tuning the dehazing (`parallel_worlds_3`)

The constructor call in `main.cpp` exposes three parameters:

```cpp
DehazeFilter dehazeFilter(defaultDevice, "src/dehazeFilters.cl",
    /* patchHalf */ 7,     // neighbourhood size = (2×7+1)² = 15×15 pixels
    /* omega     */ 0.75f, // haze removal strength: higher = more aggressive
    /* tMin      */ 0.2f   // transmission floor: lower = more amplification in darks
);
```

For genuinely hazy/foggy images, `omega = 0.95f` and `tMin = 0.1f` give stronger removal. The defaults (0.75 / 0.2) are tuned for scenes without real haze (indoor webcam).
