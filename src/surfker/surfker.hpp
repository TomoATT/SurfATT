#pragma once

#include "config.h"

#include <Eigen/Core>
#include <string>

namespace surfker {

// Diagnostics of "no zero found in fundamental mode" failures raised by the
// dispersion solver during the calls made since the last reset (per thread).
void reset_disper_diag();
int disper_fail_count();
const std::string &disper_fail_report();  // report of the first failure

struct DispersionRequest {
    Eigen::VectorX<real_t> thickness_km;
    Eigen::VectorX<real_t> depths_km;
    Eigen::VectorX<real_t> vp_km_s;
    Eigen::VectorX<real_t> vs_km_s;
    Eigen::VectorX<real_t> rho_g_cm3;
    Eigen::VectorX<real_t> periods_s;
    EarthModel earthModel;

    int iflsph = IFLSPH;  // 0: flat Earth, 1: spherical Earth
    int iwave = 2;   // 1: Love, 2: Rayleigh
    int mode = IMODE;    // 1: fundamental, 2+: higher mode
    int igr = 0;     // 0: phase velocity, >0: group velocity
};

// Compute phase/group dispersion velocities (km/s) for requested periods.
// Throws std::runtime_error on invalid input or if Fortran backend is disabled.
Eigen::VectorX<real_t> surfdisp(const DispersionRequest& req);

// Depth kernel response structure (sensitivity matrices)
struct DepthKernel1D {
    // Sensitivity matrices: dimensions are (n_periods, n_layers)
    Eigen::MatrixX<real_t> sen_vs;    // S-wave velocity sensitivity
    Eigen::MatrixX<real_t> sen_vp;    // P-wave velocity sensitivity
    Eigen::MatrixX<real_t> sen_rho;   // Density sensitivity
    Eigen::MatrixX<real_t> sen_gc;    // cos(2psi) anisotropy sensitivity
    Eigen::MatrixX<real_t> sen_gs;    // sin(2psi) anisotropy sensitivity
};

// struct DepthKernelHTI1D {
//     // Sensitivity matrices: dimensions are (n_periods, n_layers)
//     Eigen::MatrixX<real_t> sen_vs;    // S-wave velocity sensitivity
//     Eigen::MatrixX<real_t> sen_vp;    // P-wave velocity sensitivity
//     Eigen::MatrixX<real_t> sen_rho;   // Density sensitivity
//     Eigen::MatrixX<real_t> sen_gc;    // cos(2psi) anisotropy sensitivity
//     Eigen::MatrixX<real_t> sen_gs;    // sin(2psi) anisotropy sensitivity
// };

// Compute Rayleigh wave depth kernels (sensitivities) for 1D model
// Returns kernels with shape (n_periods × n_layers)
// Throws std::runtime_error on invalid input or if Fortran backend is disabled.
DepthKernel1D depthkernel1d(const DispersionRequest& req);
DepthKernel1D depthkernelHTI1d(const DispersionRequest& req);

DispersionRequest build_disp_req(const Eigen::VectorX<real_t>& dep,
                                const Eigen::VectorX<real_t>& vs,
                                const Eigen::VectorX<real_t>& periods_s,
                                int iflsph=IFLSPH, int iwave=2, int mode=IMODE, int igr=0);

// Overload: explicit Vp (km/s) and density (g/cm³) instead of empirical relations.
DispersionRequest build_disp_req(const Eigen::VectorX<real_t>& dep,
                                const Eigen::VectorX<real_t>& vs,
                                const Eigen::VectorX<real_t>& vp,
                                const Eigen::VectorX<real_t>& rho,
                                const Eigen::VectorX<real_t>& periods_s,
                                int iflsph=IFLSPH, int iwave=2, int mode=IMODE, int igr=0);
}  // namespace surfker
