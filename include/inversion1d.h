#pragma once

#include "config.h"
#include "parallel.h"
#include "input_params.h"

#include <Eigen/Core>
#include <Eigen/Dense>

class Inversion1D {
public:
    explicit Inversion1D(WaveType wavetype);
    ~Inversion1D() = default;

    Eigen::VectorX<real_t> vs1d;

    Eigen::VectorX<real_t> inv1d(
        Eigen::VectorX<real_t> zgrids,
        Eigen::VectorX<real_t> init_vs
    );

private:
    const int MAX_ITER_1D = 50;
    const real_t TOL_1D = 1e-4;
    WaveType wavetype_;

    int niter;
    std::vector<real_t> misfits;

    // Validate depth grid, initial Vs and the active src_rec tables before
    // the first forward call; logs every problem found and aborts if any.
    void check_inputs(const Eigen::VectorX<real_t>& zarr) const;

    // Abort with a diagnostic if disper() failed (returned 0 / non-finite).
    void check_pred_vel(const Eigen::VectorX<real_t>& pred_vel,
                        const Eigen::VectorX<real_t>& periods,
                        const Eigen::VectorX<real_t>& zarr,
                        int iter, WaveType wt, SurfType tp) const;
};
