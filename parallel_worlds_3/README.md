# parallel_worlds_3

An OpenCL-accelerated real-time image processing demo that runs five filters in parallel on every frame from a webcam (or a still image). Each filter is implemented as an OpenCL kernel that dispatches one GPU work-item per pixel.

## What's in this project

### Original filters (pre-existing)

| Window | Filter | What it does |
|--------|--------|--------------|
| `preview` | None | Raw camera / input image |
| `converted` | `grey` kernel | BGR → grayscale using ITU-R BT.709 luminance weights |
| `edge` | `sobel` kernel | Sobel edge detection on the grayscale image using a tiled 10×10 work-group with local memory |
| `effect` | `effectFilter` kernel | Darkens pixels (×0.5) where Sobel edge magnitude exceeds a threshold, producing a stylised look |

### Integrated: Dark Channel Prior dehazing

| Window | Filter | What it does |
|--------|--------|--------------|
| `dehazed` | DCP pipeline | Removes haze/fog from the image using the Dark Channel Prior algorithm (He et al., CVPR 2009) |

The dehazing pipeline runs as three sequential OpenCL kernel passes on the GPU, with one CPU step in between:

1. **`darkChannel` kernel** — for each pixel, finds the minimum value across all BGR channels within a (2×`patchHalf`+1)² neighbourhood.
2. **CPU: atmospheric light estimation** — reads the dark channel back to CPU, takes the top 0.1% brightest dark-channel pixels, and computes the per-channel mean of those candidates as the atmospheric light `A`. Each channel is capped at 0.85 to prevent a bright highlight (window, lamp) from causing extreme colour shifts.
3. **`estimateTransmission` kernel** — normalises each pixel by `A`, runs the same patch min, then computes `t = 1 − ω × patchMin` (ω = 0.75 by default).
4. **`recoverRadiance` kernel** — applies `J = (I − A) / max(t, tMin) + A` per pixel, clamped to [0, 1].

Default parameters: `patchHalf = 7` (15×15 patch), `omega = 0.75`, `tMin = 0.2`.

These are tuned for scenes **without real haze** (e.g. a webcam). If you have a genuinely hazy/foggy image you can increase `omega` towards 0.95 by changing the `DehazeFilter` constructor call in `main.cpp`.

### New files added

| File | Role |
|------|------|
| `src/DehazeFilter.hpp` | C++ class — subclasses `ImageFilter`, holds extra `cl::Kernel` and `cl::Buffer` members for the 3-pass pipeline |
| `src/dehazeFilters.cl` | OpenCL kernel source — `darkChannel`, `estimateTransmission`, `recoverRadiance` |

### Changes to existing files

| File | Change |
|------|--------|
| `src/main.cpp` | Added `DehazeFilter` instance, `dehazeFrame` mat, `"dehazed"` window, `--image` CLI argument |
| `src/CMakeLists.txt` | Removed `-msse2` flag (x86-only; breaks compilation on Apple Silicon ARM64) |
| `src/OpenCLInterface.hpp` | Added `CL_HPP_MINIMUM_OPENCL_VERSION 120` and `CL_HPP_TARGET_OPENCL_VERSION 120` before including `cl2.hpp` (required for macOS, which ships OpenCL 1.2 — the header defaults to 2.0 and fails to compile otherwise) |
| `src/greyImageFilters.cl` | Restored from ROT13 obfuscation to valid OpenCL source (both `grey` and `sobel` kernels) |

---

## Dependencies

| Dependency | Minimum version | Notes |
|------------|----------------|-------|
| CMake | 3.12 | |
| C++ compiler | C++11 | clang++ on macOS, g++ or clang++ on Linux, MSVC on Windows |
| OpenCV | 4.x recommended | Needs `highgui` and `imgproc` modules |
| OpenCL | 1.2 | See platform notes below |

---

## Building and running

The build system is per-subproject. All commands below are run from inside `parallel_worlds_3/src/`.

### macOS (tested — Apple Silicon M-series, macOS 14+)

OpenCL 1.2 ships with macOS via `OpenCL.framework`. No CUDA or additional OpenCL install needed. The CUDA warning during CMake configure is harmless.

**Prerequisites**

```bash
# Xcode Command Line Tools (provides clang, OpenCL headers)
xcode-select --install

# OpenCV via Homebrew
brew install opencv
```

**Build**

```bash
cd parallel_worlds_3/src
cmake -S . -B ../build
cmake --build ../build
```

**Run — live camera**

```bash
cd ../build
./parallel_worlds_3
```

**Run — still image**

```bash
cd ../build
./parallel_worlds_3 --image /path/to/your/image.jpg
```

Press any key to quit.

---

### Linux (x86_64 — not tested in this session, instructions based on code)

You need an OpenCL ICD loader and a vendor implementation (NVIDIA, AMD, or Intel).

**Prerequisites**

```bash
# Ubuntu / Debian example
sudo apt install cmake build-essential libopencv-dev ocl-icd-opencl-dev opencl-headers

# For NVIDIA GPUs, install the CUDA toolkit which includes OpenCL:
# https://developer.nvidia.com/cuda-downloads
# For AMD GPUs, install ROCm or the AMDGPU-PRO driver.
# For Intel integrated graphics, install intel-opencl-icd.
```

If CMake cannot find OpenCL automatically (because there is no CUDA toolkit), pass the path manually:

```bash
cmake -S . -B ../build -DOpenCL_ROOT=/usr
```

**Build**

```bash
cd parallel_worlds_3/src
cmake -S . -B ../build
cmake --build ../build
```

**Run — live camera**

```bash
cd ../build
./parallel_worlds_3
```

**Run — still image**

```bash
cd ../build
./parallel_worlds_3 --image /path/to/your/image.jpg
```

> **Note:** The `-msse2` flag has been removed from `CMakeLists.txt`. The build will work on both x86_64 and ARM64 Linux.

---

### Windows (not tested in this session, instructions based on code)

The `CMakeLists.txt` has a separate MSVC branch that uses `/arch:SSE2` instead of `-msse2`, so the compiler flag issue does not apply on Windows.

**Prerequisites**

- [CMake](https://cmake.org/download/) ≥ 3.12
- Visual Studio 2019 or 2022 with the "Desktop development with C++" workload
- [OpenCV Windows release](https://opencv.org/releases/) — set `OpenCV_DIR` in CMake to the build directory containing `OpenCVConfig.cmake`
- OpenCL — the easiest way is to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (includes OpenCL headers and `OpenCL.lib`) for NVIDIA GPUs. For AMD or Intel, install the respective GPU drivers which include an OpenCL ICD.

**Build (Developer Command Prompt)**

```bat
cd parallel_worlds_3\src
cmake -S . -B ..\build -DOpenCV_DIR="C:\path\to\opencv\build"
cmake --build ..\build --config Release
```

**Run — live camera**

```bat
cd ..\build\Release
parallel_worlds_3.exe
```

**Run — still image**

```bat
parallel_worlds_3.exe --image C:\path\to\your\image.jpg
```

> **Note:** On Windows the executable is placed in `build\Release\` or `build\Debug\` depending on the build config. The post-build step copies `src\` and `preview.png` alongside it automatically.

---

## Input fallback order

When no `--image` argument is given:

1. Tries to open webcam device `0`
2. If no camera is found, reads `preview.png` from the current working directory
3. If neither is available, exits with code 3

The program must be run from the `build/` directory (not `src/`) because the OpenCL kernel files are loaded at runtime from the relative path `src/*.cl`, and CMake copies the entire `src/` tree into `build/src/` as a post-build step.

---

## Tuning the dehazing

The `DehazeFilter` constructor in `main.cpp` accepts three optional parameters:

```cpp
DehazeFilter dehazeFilter(defaultDevice, "src/dehazeFilters.cl",
    /* patchHalf */ 7,     // patch window = (2*7+1)² = 15×15 pixels
    /* omega     */ 0.75f, // haze retention: higher = more aggressive removal
    /* tMin      */ 0.2f   // transmission floor: lower = more amplification in dark areas
);
```

For genuinely hazy/foggy images, try `omega = 0.95f` and `tMin = 0.1f`. For indoor/non-hazy scenes, keep the defaults.
