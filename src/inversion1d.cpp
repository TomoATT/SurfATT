#include "inversion1d.h"
#include "logger.h"
#include "surfker/surfker.hpp"
#include "utils.h"
#include "input_params.h"
#include "src_rec.h"
#include <stdexcept>

Inversion1D::Inversion1D(WaveType wavetype)
    : wavetype_(wavetype) {
    niter = 0;
    misfits.clear();
    vs1d.resize(0);
}

Eigen::VectorX<real_t> Inversion1D::inv1d(
    Eigen::VectorX<real_t> zarr,
    Eigen::VectorX<real_t> init_vs
) {
    auto& IP = InputParams::IP();
    auto& logger = ATTLogger::logger();

    vs1d = init_vs;
    misfits.clear();
    niter = 0;

    int nz = static_cast<int>(zarr.size());
    real_t step_length = IP.inversion().step_length;

    real_t sigma = _0_CR;
    if (IP.postproc().smooth_method == 0) {
        sigma = IP.postproc().sigma[1];
    } else if (IP.postproc().smooth_method == 1) {
        sigma = 0.68 * (zarr(zarr.size() - 1) - zarr(0)) / IP.postproc().n_inv_grid[2];
    }
    int n_active_for_wave = 0;
    for (const auto& [wt, tp] : IP.data().active_data) {
        (void)tp;
        if (wt == wavetype_) {
            ++n_active_for_wave;
        }
    }
    if (n_active_for_wave == 0) {
        throw std::runtime_error(
            "Inversion1D::inv1d: selected wavetype has no active surface-wave data");
    }

    logger.Info(
        fmt::format("1D inversion using averaged {} surface-wave data",
            waveTypeStr[static_cast<int>(wavetype_)]),
        MODULE_INV1D
    );
    logger.Debug(
        fmt::format("  nz={}, sigma={:.3f}, initial step_length={:.3e}, max_iter={}",
            nz, sigma, step_length, MAX_ITER_1D),
        MODULE_INV1D
    );
    logger.Debug(
        fmt::format("  Initial Vs: min={:.4f}, max={:.4f} km/s",
            vs1d.minCoeff(), vs1d.maxCoeff()),
        MODULE_INV1D
    );

    // define model update vector
    Eigen::VectorX<real_t> update(nz);
    Eigen::VectorX<real_t> update_total(nz);

    int iter = 0;
    for (iter = 0; iter < MAX_ITER_1D; ++iter) {
        // Compute predicted dispersion curve and misfit
        update_total.setZero();
        real_t misfit_total = _0_CR;
        for (auto [wt, tp] : IP.data().active_data) {
            if (wt != wavetype_) continue;
            int itype = static_cast<int>(tp);
            auto &sr = SrcRec::SR(wt, tp);
            int nperiod = sr.periods_info.nperiod;

            surfker::DispersionRequest req = surfker::build_disp_req(
                zarr, vs1d, sr.periods_info.periods,
                IFLSPH, iwave_of(wt), IMODE, itype
            );

            surfker::reset_disper_diag();
            Eigen::VectorX<real_t> pred_vel = surfker::surfdisp(req);
            if (surfker::disper_fail_count() > 0) {
                logger.Warn(fmt::format(
                    "1D inversion iter {}: dispersion calculation failed for {}_{} "
                    "(Vs range [{:.4f}, {:.4f}] km/s)\n{}",
                    iter, waveTypeStr[static_cast<int>(wt)], (itype == 0 ? "PH" : "GR"),
                    vs1d.minCoeff(), vs1d.maxCoeff(), surfker::disper_fail_report()),
                    MODULE_INV1D);
            }
            real_t misfit = 0.5 * (pred_vel - sr.periods_info.meanvel).array().square().sum();
            logger.Debug(
                fmt::format("  iter {:3d} | {}_{} misfit={:.6e} (weight={:.3f})",
                    iter, waveTypeStr[static_cast<int>(wt)],
                    (itype == 0 ? "PH" : "GR"), misfit, IP.data().weights[itype]),
                MODULE_INV1D
            );
            misfit_total += misfit * IP.data().weights[itype];

            surfker::DepthKernel1D kernels = surfker::depthkernel1d(req);

            update.setZero();
            auto vp = vs2vp<real_t>(vs1d);
            auto db = dalpha_dbeta<real_t>(vs1d);
            auto dr = drho_dalpha<real_t>(vp);
            // Love-wave depth kernels carry no vp sensitivity, so sen_vp stays
            // an empty matrix; guard each chain-rule term by its kernel size.
            const bool has_vp  = kernels.sen_vp.size()  > 0;
            const bool has_rho = kernels.sen_rho.size() > 0;
            for (int iper = 0; iper < nperiod; ++iper) {
                Eigen::VectorX<real_t> sen = kernels.sen_vs.row(iper).transpose();
                if (has_vp) {
                    sen.array() += kernels.sen_vp.row(iper).transpose().array() * db.array();
                }
                if (has_rho) {
                    sen.array() += kernels.sen_rho.row(iper).transpose().array() * dr.array() * db.array();
                }
                update += sen * (pred_vel(iper) - sr.periods_info.meanvel(iper));
            }
            update /= nperiod;
            update = gaussian_smooth_1d(update, zarr, sigma);
            update_total += update * IP.data().weights[itype];
        }
        misfits.push_back(misfit_total);

        if (iter > 0 && misfits[iter] > misfits[iter - 1]) {
            step_length *= IP.inversion().maxshrink;
        }
        update_total = step_length * update_total / update_total.lpNorm<Eigen::Infinity>();

        logger.Debug(
            fmt::format("Iteration {}: misfit = {:.6e}, step_length = {:.3e}", iter, misfits.back(), step_length),
            MODULE_INV1D
        );

        if (iter > 0) {
            real_t derr = std::abs(misfits[iter] - misfits[iter - 1]);
            logger.Debug(
                fmt::format("  iter {:3d} | delta_misfit={:.3e} (tol={:.3e})",
                    iter, derr, TOL_1D),
                MODULE_INV1D
            );
            if (derr < TOL_1D) {
                vs1d -= update_total;
                break;
            }
        }
        vs1d -= update_total;
    }
    niter = iter;

    if (niter < MAX_ITER_1D) {
        logger.Info(
            fmt::format("1D inversion converged after {} iterations, final misfit={:.6e}",
                niter + 1, misfits.back()),
            MODULE_INV1D
        );
    } else {
        logger.Warn(
            fmt::format("1D inversion reached max iterations ({}), final misfit={:.6e}",
                MAX_ITER_1D, misfits.back()),
            MODULE_INV1D
        );
    }
    logger.Debug(
        fmt::format("  Final Vs: min={:.4f}, max={:.4f} km/s",
            vs1d.minCoeff(), vs1d.maxCoeff()),
        MODULE_INV1D
    );

    return vs1d;
}
