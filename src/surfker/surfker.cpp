#include "utils.h"
#include "surfdisp.hpp"
#include "surfker.hpp"
#include <Eigen/Core>
#include <stdexcept>

namespace {

void validate_request(const surfker::DispersionRequest& req) {
    const auto nlayer = req.thickness_km.size();
    if (nlayer == 0) {
        throw std::runtime_error("surfker::surfdisp: model is empty");
    }
    if (req.vp_km_s.size() != nlayer || req.vs_km_s.size() != nlayer || req.rho_g_cm3.size() != nlayer) {
        throw std::runtime_error("surfker::surfdisp: model vectors must have same length");
    }
    if (req.periods_s.size() == 0) {
        throw std::runtime_error("surfker::surfdisp: periods_s is empty");
    }
    if (req.mode < 1) {
        throw std::runtime_error("surfker::surfdisp: mode must be >= 1");
    }
    if (req.iwave != 1 && req.iwave != 2) {
        throw std::runtime_error("surfker::surfdisp: iwave must be 1 (Love) or 2 (Rayleigh)");
    }
}

surfker::DispersionRequest refine_request(
    const Eigen::VectorX<real_t>& dep,
    const Eigen::VectorX<real_t>& vs,
    const Eigen::VectorX<real_t>& vp,
    const Eigen::VectorX<real_t>& rho,
    const Eigen::VectorX<real_t>& periods_s,
    int iflsph, int iwave, int mode, int igr) {

    if (dep.size() != vs.size() || dep.size() != vp.size() || dep.size() != rho.size()) {
        throw std::runtime_error("build_disp_req: dep, vs, vp, and rho must have the same length");
    }

    const int nz = static_cast<int>(dep.size());
    const int mmax = nz + 1;

    surfker::DispersionRequest req;
    req.iflsph = iflsph;
    req.earthModel = (iflsph == 1) ? EarthModel::Spherical : EarthModel::Flat;  // hardcoded for now; can be made user-configurable if needed
    req.iwave = iwave;
    req.mode = mode;
    req.igr = igr;

    req.depths_km = Eigen::VectorX<real_t>::Zero(mmax);
    req.thickness_km = Eigen::VectorX<real_t>::Zero(mmax);
    req.vs_km_s = Eigen::VectorX<real_t>::Zero(mmax);
    req.vp_km_s = Eigen::VectorX<real_t>::Zero(mmax);
    req.rho_g_cm3 = Eigen::VectorX<real_t>::Zero(mmax);
    req.periods_s = periods_s;

    req.depths_km.head(nz) = dep;
    req.depths_km(mmax - 1) = dep(nz - 1);

    for (int kk = 0; kk < nz; ++kk) {
        req.vs_km_s(kk) = vs(kk);
        req.vp_km_s(kk) = vp(kk);
        req.rho_g_cm3(kk) = rho(kk);
        if (kk == nz - 1) {
            req.thickness_km(kk) = (nz > 1) ? req.thickness_km(kk - 1) : _0_CR;
        } else {
            req.thickness_km(kk) = dep(kk + 1) - dep(kk);
        }
    }

    req.thickness_km(mmax - 1) = _0_CR;
    req.vp_km_s(mmax - 1) = req.vp_km_s(nz - 1);
    req.vs_km_s(mmax - 1) = req.vs_km_s(nz - 1);
    req.rho_g_cm3(mmax - 1) = req.rho_g_cm3(nz - 1);

    return req;
}

}  // anonymous namespace

namespace surfker {

void reset_disper_diag() { disper_diag_reset(); }
int disper_fail_count() { return disper_diag_nfail(); }
const std::string &disper_fail_report() { return disper_diag_report(); }

DispersionRequest build_disp_req(const Eigen::VectorX<real_t>& dep,
                                const Eigen::VectorX<real_t>& vs,
                                const Eigen::VectorX<real_t>& periods_s,
                                int iflsph, int iwave, int mode, int igr) {

    if (dep.size() != vs.size()) {
        throw std::runtime_error("build_disp_req: depths_km and vs_km_s must have same length");
    }
    const Eigen::VectorX<real_t> vp = vs2vp<real_t>(vs);
    const Eigen::VectorX<real_t> rho = vp2rho<real_t>(vp);
    return refine_request(dep, vs, vp, rho, periods_s,
                          iflsph, iwave, mode, igr);

}  // build_disp_req

DispersionRequest build_disp_req(const Eigen::VectorX<real_t>& dep,
                                const Eigen::VectorX<real_t>& vs,
                                const Eigen::VectorX<real_t>& vp,
                                const Eigen::VectorX<real_t>& rho,
                                const Eigen::VectorX<real_t>& periods_s,
                                int iflsph, int iwave, int mode, int igr) {

    return refine_request(dep, vs, vp, rho, periods_s,
                          iflsph, iwave, mode, igr);

}  // build_disp_req (vp, rho overload)

// Helper function: Compute Rayleigh wave phase velocity depth kernels
DepthKernel1D depthkernel_rayleigh_phase(const DispersionRequest& req) {
    const int kmax = static_cast<int>(req.periods_s.size());
    const int mmax = static_cast<int>(req.vs_km_s.size());
    const int nz = mmax - 1;

    // Type conversion from req data
    Eigen::VectorXf rthk = req.thickness_km.cast<float>();
    Eigen::VectorXf rvp  = req.vp_km_s.cast<float>();
    Eigen::VectorXf rvs  = req.vs_km_s.cast<float>();
    Eigen::VectorXf rrho = req.rho_g_cm3.cast<float>();
    Eigen::VectorXd t = req.periods_s.cast<double>();

    // Compute phase velocities
    std::vector<double> cp = disper(
        rthk.data(), rvp.data(), rvs.data(), rrho.data(),
        mmax, req.iflsph, req.iwave, req.mode, 0, kmax, t.data()
    );

    Eigen::MatrixX<real_t> sen_vs = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_vp = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_rho = Eigen::MatrixX<real_t>::Zero(kmax, nz);

    for (int i = 0; i < kmax; ++i) {

        double t_val = t(i);
        double cp_val = cp[i];
        RayleighEigenResult result =rayleighPhaseKernel(
            mmax, rthk.data(), rvp.data(), rvs.data(), rrho.data(),
            t_val, cp_val, req.earthModel
        );

        // Extract kernel for this period (first nz layers, skip halfspace)
        for (int j = 0; j < nz; ++j) {
            sen_vs(i, j) = static_cast<real_t>(result.dc2db[j]);
            sen_vp(i, j) = static_cast<real_t>(result.dc2da[j]);
            sen_rho(i, j) = static_cast<real_t>(result.dc2dr[j]);
        }
    }
    DepthKernel1D kernels;
    kernels.sen_vs = sen_vs;
    kernels.sen_vp = sen_vp;
    kernels.sen_rho = sen_rho;

    return kernels;
}

// Helper function: Compute Rayleigh wave group velocity depth kernels
DepthKernel1D depthkernel_rayleigh_group(const DispersionRequest& req) {
    const int kmax = static_cast<int>(req.periods_s.size());
    const int mmax = static_cast<int>(req.vs_km_s.size());
    const int nz = mmax - 1;

    // Type conversion from req data
    Eigen::VectorXf rthk = req.thickness_km.cast<float>();
    Eigen::VectorXf rvp  = req.vp_km_s.cast<float>();
    Eigen::VectorXf rvs  = req.vs_km_s.cast<float>();
    Eigen::VectorXf rrho = req.rho_g_cm3.cast<float>();
    Eigen::VectorXd t = req.periods_s.cast<double>();

    // Compute phase velocities at base period
    std::vector<double> cp = disper(
        rthk.data(), rvp.data(), rvs.data(), rrho.data(),
        mmax, req.iflsph, req.iwave, req.mode, 0, kmax, t.data()
    );

    Eigen::MatrixX<real_t> sen_vs = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_vp = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_rho = Eigen::MatrixX<real_t>::Zero(kmax, nz);

    // Rayleigh group velocity kernels (finite-difference derivatives)
    double dt = 0.01;  // Fractional period increment
    Eigen::VectorXd t1(kmax), t2(kmax);
    for (int i = 0; i < kmax; ++i) {
        t1(i) = t(i) * (1.0 + 0.5 * dt);
        t2(i) = t(i) * (1.0 - 0.5 * dt);
    }

    // Compute phase velocities at perturbed periods
    std::vector<double> cp1 = disper(
        rthk.data(), rvp.data(), rvs.data(), rrho.data(),
        mmax, req.iflsph, req.iwave, req.mode, 0, kmax, t1.data()
    );
    std::vector<double> cp2 = disper(
        rthk.data(), rvp.data(), rvs.data(), rrho.data(),
        mmax, req.iflsph, req.iwave, req.mode, 0, kmax, t2.data()
    );

    for (int i = 0; i < kmax; ++i) {
        double t_val = t(i);
        double cp_val = cp[i];
        double t1_val = t1(i);
        double cp1_val = cp1[i];
        double t2_val = t2(i);
        double cp2_val = cp2[i];

        RayleighGroupKernelResult result = rayleighGroupKernel(
            mmax, rthk.data(), rvp.data(), rvs.data(), rrho.data(),
            t_val, cp_val, t1_val, cp1_val, t2_val, cp2_val, req.earthModel
        );

        // Extract group velocity kernels (first nz layers, skip halfspace)
        for (int j = 0; j < nz; ++j) {
            sen_vs(i, j) = static_cast<real_t>(result.du2db[j]);
            sen_vp(i, j) = static_cast<real_t>(result.du2da[j]);
            sen_rho(i, j) = static_cast<real_t>(result.du2dr[j]);
        }
    }

    DepthKernel1D kernels;
    kernels.sen_vs = sen_vs;
    kernels.sen_vp = sen_vp;
    kernels.sen_rho = sen_rho;

    return kernels;
}

DepthKernel1D depthkernel_love_phase(const DispersionRequest& req) {
    const int kmax = static_cast<int>(req.periods_s.size());
    const int mmax = static_cast<int>(req.vs_km_s.size());
    const int nz = mmax - 1;

    // Type conversion from req data
    Eigen::VectorXf rthk = req.thickness_km.cast<float>();
    Eigen::VectorXf rvp  = req.vp_km_s.cast<float>();
    Eigen::VectorXf rvs  = req.vs_km_s.cast<float>();
    Eigen::VectorXf rrho = req.rho_g_cm3.cast<float>();
    Eigen::VectorXd t = req.periods_s.cast<double>();

    // First compute Love wave phase velocities for all periods
    std::vector<double> cp = disper(
        rthk.data(), rvp.data(), rvs.data(), rrho.data(),
        mmax, req.iflsph, 1, req.mode, 0, kmax, t.data()
    );

    Eigen::MatrixX<real_t> sen_vs = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_rho = Eigen::MatrixX<real_t>::Zero(kmax, nz);

    // Compute Love wave depth kernels for each period
    for (int i = 0; i < kmax; ++i) {
        double t_val = t(i);
        double cp_val = cp[i];

        LoveEigenResult result = lovePhaseKernel(
            mmax, rthk.data(), rvs.data(), rrho.data(),
            t_val, cp_val, req.earthModel
        );

        // Extract kernel for this period (first nz layers, skip halfspace)
        for (int j = 0; j < nz; ++j) {
            sen_vs(i, j) = static_cast<real_t>(result.dc2db[j]);
            sen_rho(i, j) = static_cast<real_t>(result.dc2dr[j]);
        }
    }

    DepthKernel1D kernels;
    kernels.sen_vs = sen_vs;
    kernels.sen_rho = sen_rho;

    return kernels;
}

DepthKernel1D depthkernel_love_group(const DispersionRequest& req) {
    const int kmax = static_cast<int>(req.periods_s.size());
    const int mmax = static_cast<int>(req.vs_km_s.size());
    const int nz = mmax - 1;

    Eigen::VectorXf rthk = req.thickness_km.cast<float>();
    Eigen::VectorXf rvp  = req.vp_km_s.cast<float>();
    Eigen::VectorXf rvs  = req.vs_km_s.cast<float>();
    Eigen::VectorXf rrho = req.rho_g_cm3.cast<float>();
    Eigen::VectorXd t = req.periods_s.cast<double>();

    // Match legacy implementation: +/-5% period perturbation for Love group kernels.
    Eigen::VectorXd t1(kmax), t2(kmax);
    double dt = 0.01;  // Fractional period increment
    for (int i = 0; i < kmax; ++i) {
        t1(i) = t(i) * (1.0 + 0.5 * dt);
        t2(i) = t(i) * (1.0 - 0.5 * dt);
    }

    std::vector<double> cp = disper(
        rthk.data(), rvp.data(), rvs.data(), rrho.data(),
        mmax, req.iflsph, 1, req.mode, 0, kmax, t.data()
    );
    std::vector<double> cp1 = disper(
        rthk.data(), rvp.data(), rvs.data(), rrho.data(),
        mmax, req.iflsph, 1, req.mode, 0, kmax, t1.data()
    );
    std::vector<double> cp2 = disper(
        rthk.data(), rvp.data(), rvs.data(), rrho.data(),
        mmax, req.iflsph, 1, req.mode, 0, kmax, t2.data()
    );

    Eigen::MatrixX<real_t> sen_vs = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_rho = Eigen::MatrixX<real_t>::Zero(kmax, nz);

    for (int i = 0; i < kmax; ++i) {
        double t_val = t(i);
        double cp_val = cp[i];
        double t1_val = t1(i);
        double cp1_val = cp1[i];
        double t2_val = t2(i);
        double cp2_val = cp2[i];

        LoveGroupKernelResult result = loveGroupKernel(
            mmax, rthk.data(), rvs.data(), rrho.data(),
            t_val, cp_val, t1_val, cp1_val,
            t2_val, cp2_val, req.earthModel
        );

        for (int j = 0; j < nz; ++j) {
            sen_vs(i, j) = static_cast<real_t>(result.du2db[j]);
            sen_rho(i, j) = static_cast<real_t>(result.du2dr[j]);
        }
    }

    DepthKernel1D kernels;
    kernels.sen_vs = sen_vs;
    kernels.sen_rho = sen_rho;

    return kernels;
}

Eigen::VectorX<real_t> surfdisp(const DispersionRequest& req) {
    validate_request(req);

    const int nlayer = static_cast<int>(req.thickness_km.size());
    const int kmax = static_cast<int>(req.periods_s.size());

    Eigen::VectorXf thkm(nlayer);
    Eigen::VectorXf vpm(nlayer);
    Eigen::VectorXf vsm(nlayer);
    Eigen::VectorXf rhom(nlayer);

    for (int i = 0; i < nlayer; ++i) {
        thkm(i) = static_cast<float>(req.thickness_km(i));
        vpm(i) = static_cast<float>(req.vp_km_s(i));
        vsm(i) = static_cast<float>(req.vs_km_s(i));
        rhom(i) = static_cast<float>(req.rho_g_cm3(i));
    }

    Eigen::VectorX<real_t> t(kmax);
    for (int i = 0; i < kmax; ++i) {
        t(i) = static_cast<real_t>(req.periods_s(i));
    }

    std::vector<double> cg = disper(
        thkm.data(), vpm.data(), vsm.data(), rhom.data(),
        nlayer, req.iflsph, req.iwave, req.mode, req.igr, kmax, t.data()
    );

    Eigen::VectorX<real_t> out(kmax);
    for (int i = 0; i < kmax; ++i) {
        out(i) = static_cast<real_t>(cg[i]);
    }
    return out;

}

DepthKernel1D depthkernel1d(const DispersionRequest& req) {
    validate_request(req);

    if (req.iwave == 2 && req.igr == 0) {
        return depthkernel_rayleigh_phase(req);
    } else if (req.iwave == 2 && req.igr == 1) {
        return depthkernel_rayleigh_group(req);
    } else if (req.iwave == 1 && req.igr == 0) {
        return depthkernel_love_phase(req);
    } else if (req.iwave == 1 && req.igr == 1) {
        return depthkernel_love_group(req);
    } else {
        throw std::runtime_error(
            "surfker::depthkernel1d: Only Love/Rayleigh waves with igr=0 (phase) or igr=1 (group) are supported");
    }
}

DepthKernel1D depthkernelHTI1d(const DispersionRequest& req) {
    validate_request(req);

    if (!(req.iwave == 2 && req.igr == 0)) {
        throw std::runtime_error(
            "surfker::depthkernelHTI1d: only Rayleigh phase velocity is supported (iwave=2, igr=0)");
    }

    const int mmax = static_cast<int>(req.vs_km_s.size());
    const int kmax = static_cast<int>(req.periods_s.size());
    const int nz = mmax - 1;

    Eigen::VectorXf rthk = req.thickness_km.cast<float>();
    Eigen::VectorXf rvp  = req.vp_km_s.cast<float>();
    Eigen::VectorXf rvs  = req.vs_km_s.cast<float>();
    Eigen::VectorXf rrho = req.rho_g_cm3.cast<float>();

    Eigen::VectorXd t = req.periods_s.cast<double>();
    std::vector<double> cp = disper(
        rthk.data(), rvp.data(), rvs.data(), rrho.data(),
        mmax, req.iflsph, req.iwave, req.mode, 0, kmax, t.data()
    );

    Eigen::MatrixX<real_t> sen_vs  = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_vp  = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_rho = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_gc  = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_gs  = Eigen::MatrixX<real_t>::Zero(kmax, nz);

    for (int i = 0; i < kmax; ++i) {
        double t_val = t(i);
        double cp_val = cp[i];
        HTIResult result = rayleighPhaseKernel_hti(
            mmax, rthk.data(), rvp.data(), rvs.data(), rrho.data(),
            t_val, cp_val, req.earthModel
        );

        for (int j = 0; j < nz; ++j) {
            sen_vs(i, j) = static_cast<real_t>(result.dc2db[j]);
            sen_vp(i, j) = static_cast<real_t>(result.dc2da[j]);
            sen_rho(i, j) = static_cast<real_t>(result.dc2dr[j]);
            sen_gc(i, j) = static_cast<real_t>(result.dc2dgc[j]);
            sen_gs(i, j) = static_cast<real_t>(result.dc2dgs[j]);
        }
    }

    return DepthKernel1D{sen_vs, sen_vp, sen_rho, sen_gc, sen_gs};
}

}  // namespace surfker
