# Lidar to camera projection of KITTI using oneMath

Fork of the [azureology/kitti-velo2cam](https://github.com/azureology/kitti-velo2cam)
demo with a version where the main transformation computation is offloaded using
[oneMath](https://github.com/uxlfoundation/oneMath). See the original repo for
context and details.

Dependencies:
* a SYCL compiler (tested with DPC++ only)
* a build of oneMath with the backends of choice (tested MKL and cuBLAS backends)
* pybind11
* numpy

Steps to run:
* Compile with `make` in the main directory. Use `make CXX=clang++` if using open-source DPC++.
* Run with `ONEAPI_DEVICE_SELECTOR=<value> python proj_velo2cam_onemath.py` where `<value>` is the offload device of choice.
* Similarly, run benchmarking with `ONEAPI_DEVICE_SELECTOR=<value> python bench.py.
It takes two optional command-line arguments. The first one is the number of points
to transform. If a second argument `cm` is added, only the col_major implementation
is benchmarked. This is to facilitate running on backends which don't support
row_major like cuBLAS.
