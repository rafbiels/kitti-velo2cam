#include "oneapi/math.hpp"
#include <sycl/sycl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <optional>

namespace onemath_v2c {
namespace detail {
    static std::optional<sycl::queue> defaultQueue;
}
sycl::queue& getDefaultQueue() {
    return detail::defaultQueue.has_value() ?
        detail::defaultQueue.value() :
        detail::defaultQueue.emplace(
            #ifdef IN_ORDER_QUEUE
            sycl::property::queue::in_order{}
            #endif
    );
}

namespace py = pybind11;
using py_array = py::array_t<float, py::array::c_style | py::array::forcecast>;

py_array velo2cam_rm(py_array velo, py_array trf, py_array rect, py_array p2) {
    py_array output{{p2.shape(0), velo.shape(1)}};
    sycl::buffer<float,1> buf_output{output.mutable_data(0), output.size()};
    sycl::buffer<float,1> buf_tmp1{trf.shape(0)*velo.shape(1)};
    sycl::buffer<float,1> buf_tmp2{rect.shape(0)*velo.shape(1)};
    sycl::buffer<float,1> buf_velo{velo.data(0), velo.size()};
    sycl::buffer<float,1> buf_trf{trf.data(0), trf.size()};
    sycl::buffer<float,1> buf_rect{rect.data(0), rect.size()};
    sycl::buffer<float,1> buf_p2{p2.data(0), p2.size()};
    using oneapi::math::blas::row_major::gemm;
    auto nontrans = oneapi::math::transpose::nontrans;
    sycl::queue& q{getDefaultQueue()};
    gemm(q, nontrans, nontrans,
         trf.shape(0), velo.shape(1), trf.shape(1),
         1.0f,
         buf_trf,
         trf.shape(1),
         buf_velo,
         velo.shape(1),
         0.0f,
         buf_tmp1,
         velo.shape(1));
    gemm(q, nontrans, nontrans,
         rect.shape(0), velo.shape(1), rect.shape(1),
         1.0f,
         buf_rect,
         rect.shape(1),
         buf_tmp1,
         velo.shape(1),
         0.0f,
         buf_tmp2,
         velo.shape(1));
    gemm(q, nontrans, nontrans,
         p2.shape(0), velo.shape(1), p2.shape(1),
         1.0f,
         buf_p2,
         p2.shape(1),
         buf_tmp2,
         velo.shape(1),
         0.0f,
         buf_output,
         output.shape(1));
    return output;
}

py_array velo2cam_cm(py_array velo, py_array trf, py_array rect, py_array p2) {
    py_array output{{p2.shape(0), velo.shape(1)}};
    sycl::buffer<float,1> buf_output{output.mutable_data(0), output.size()};
    sycl::buffer<float,1> buf_tmp1{trf.shape(0)*velo.shape(1)};
    sycl::buffer<float,1> buf_tmp2{rect.shape(0)*velo.shape(1)};
    sycl::buffer<float,1> buf_tmp3{p2.shape(0)*velo.shape(1)};
    sycl::buffer<float,1> buf_velo{velo.data(0), velo.size()};
    sycl::buffer<float,1> buf_trf{trf.data(0), trf.size()};
    sycl::buffer<float,1> buf_rect{rect.data(0), rect.size()};
    sycl::buffer<float,1> buf_p2{p2.data(0), p2.size()};
    using oneapi::math::blas::column_major::gemm;
    using oneapi::math::blas::column_major::omatcopy;
    auto nontrans = oneapi::math::transpose::nontrans;
    auto trans = oneapi::math::transpose::trans;
    sycl::queue& q{getDefaultQueue()};
    gemm(q, trans, trans,
         trf.shape(0), velo.shape(1), trf.shape(1),
         1.0f,
         buf_trf,
         trf.shape(1),
         buf_velo,
         velo.shape(1),
         0.0f,
         buf_tmp1,
         trf.shape(0));
    gemm(q, trans, nontrans,
         rect.shape(0), velo.shape(1), rect.shape(1),
         1.0f,
         buf_rect,
         rect.shape(1),
         buf_tmp1,
         trf.shape(0),
         0.0f,
         buf_tmp2,
         rect.shape(0));
    gemm(q, trans, nontrans,
         p2.shape(0), velo.shape(1), p2.shape(1),
         1.0f,
         buf_p2,
         p2.shape(1),
         buf_tmp2,
         rect.shape(0),
         0.0f,
         buf_tmp3,
         p2.shape(0));
    omatcopy(q, trans,
             p2.shape(0), velo.shape(1),
             1.0f,
             buf_tmp3,
             p2.shape(0),
             buf_output,
             velo.shape(1));
    return output;
}
} // namespace onemath_v2c

PYBIND11_MODULE(onemath_v2c, m) {
    m.doc() = "velo2cam transformation implemented with oneMath";

    m.def("velo2cam_rm", &onemath_v2c::velo2cam_rm, "velo2cam transformation");
    m.def("velo2cam_cm", &onemath_v2c::velo2cam_cm, "velo2cam transformation");
    m.def("velo2cam", &onemath_v2c::velo2cam_cm, "velo2cam transformation");
}
