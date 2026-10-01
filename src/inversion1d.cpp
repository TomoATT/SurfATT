#include "inversion1d.h"
#include "logger.h"
#include "surfker/surfker.hpp"
#include "utils.h"
#include "input_params.h"
#include "src_rec.h"
#include "parallel.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

// inv1d runs redundantly on every rank with identical inputs, so all ranks
// reach the same failure. Only rank 0 logs; the barrier keeps the other ranks'
// MPI_Abort from killing rank 0 before its diagnostics are flushed.
[[noreturn]] void abort_all_ranks() {
    auto& mpi = Parallel::mpi();
    mpi.barrier();
    mpi.abort(EXIT_FAILURE);
}

// Index of the first row containing inf/NaN, or -1 if all entries are finite.
// (Unlike lpNorm/maxCoeff, allFinite() is guaranteed to see every NaN.)
template <typename Derived>
int first_nonfinite_row(const Eigen::DenseBase<Derived>& m) {
    for (Eigen::Index i = 0; i < m.rows(); ++i) {
        if (!m.row(i).allFinite()) return static_cast<int>(i);
    }
    return -1;
}

std::string data_tag(WaveType wt, SurfType tp) {
    return waveTypeStr[static_cast<int>(wt)] + (tp == SurfType::PH ? "_PH" : "_GR");
}

}  // namespace

Inversion1D::Inversion1D(WaveType wavetype)
    : wavetype_(wavetype) {
    niter = 0;
    misfits.clear();
    vs1d.resize(0);
}

void Inversion1D::check_inputs(const Eigen::VectorX<real_t>& zarr) const {
    auto& IP = InputParams::IP();
    auto& logger = ATTLogger::logger();
    int n_err = 0;
    auto error = [&](const std::string& msg) {
        logger.Error(msg, MODULE_INV1D);
        ++n_err;
    };

    // 1. Depth grid: finite and strictly increasing (layer thickness > 0)
    const int nz = static_cast<int>(zarr.size());
    if (nz < 2) {
        error(fmt::format("Depth grid has {} node(s); need >= 2. Check domain.depth_min_max / interval.", nz));
    }
    for (int k = 0; k < nz; ++k) {
        if (!std::isfinite(zarr(k))) {
            error(fmt::format("Depth grid node {} is non-finite ({})", k, zarr(k)));
        } else if (k > 0 && !(zarr(k) > zarr(k - 1))) {
            error(fmt::format("Depth grid not strictly increasing at node {}: z={} <= z[{}]={}",
                k, zarr(k), k - 1, zarr(k - 1)));
        }
    }

    // 2. Initial Vs: finite and positive at every node
    for (int k = 0; k < vs1d.size(); ++k) {
        if (!std::isfinite(vs1d(k)) || vs1d(k) <= _0_CR) {
            error(fmt::format("Initial Vs invalid at node {} (z={} km): Vs={}. Check model.vel_range.",
                k, k < nz ? zarr(k) : NAN, vs1d(k)));
        }
    }

    // 3. Active src_rec tables for this wave type
    for (auto [wt, tp] : IP.data().active_data) {
        if (wt != wavetype_) continue;
        auto& sr = SrcRec::SR(wt, tp);
        const std::string tag = data_tag(wt, tp);
        const std::string& file = IP.data().file_of(wt, tp);

        // Periods are validated by SrcRec::load in every run mode; vel only
        // when inverting, but inv1d fits vel in forward-only runs too.
        if (sr.check_positive(sr.vel, "vel (or dist/tt)") > 0) ++n_err;

        const auto& pinfo = sr.periods_info;
        if (pinfo.nperiod <= 0) {
            error(fmt::format("[{}] {}: no periods found (empty table?)", tag, file));
            continue;
        }
        for (int ip = 0; ip < pinfo.nperiod; ++ip) {
            if (!std::isfinite(pinfo.meanvel(ip)) || pinfo.meanvel(ip) <= _0_CR) {
                error(fmt::format("[{}] {}: mean velocity at period {} s is invalid ({})",
                    tag, file, pinfo.periods(ip), pinfo.meanvel(ip)));
            }
            // Distinct values closer than real_t_equal's tolerance are split
            // into separate periods by get_periods(); usually a formatting issue.
            if (ip > 0 && std::isfinite(pinfo.periods(ip)) &&
                real_t_equal(pinfo.periods(ip), pinfo.periods(ip - 1))) {
                logger.Warn(fmt::format("[{}] {}: near-duplicate periods {} and {} are treated as different periods",
                    tag, file, pinfo.periods(ip - 1), pinfo.periods(ip)), MODULE_INV1D);
            }
        }
        std::string plist;
        for (int ip = 0; ip < pinfo.nperiod; ++ip) plist += fmt::format(" {}", pinfo.periods(ip));
        logger.Debug(fmt::format("  [{}] {} rows, {} periods:{}", tag, sr.n_obs(), pinfo.nperiod, plist),
            MODULE_INV1D);
    }

    if (n_err > 0) {
        logger.Error(fmt::format("1D inversion input check failed with {} error(s); see messages above.", n_err),
            MODULE_INV1D);
        abort_all_ranks();
    }
}

void Inversion1D::log_vs_profile(const Eigen::VectorX<real_t>& zarr, int n_update) const {
    auto& logger = ATTLogger::logger();
    // minCoeff/maxCoeff are unreliable with NaN, so take the range over finite
    // nodes and count the rest separately.
    std::string prof;
    real_t vmin = std::numeric_limits<real_t>::infinity();
    real_t vmax = -vmin;
    int n_nonfinite = 0;
    for (int k = 0; k < vs1d.size(); ++k) {
        prof += fmt::format(" {:.2f}:{:.4f}", zarr(k), vs1d(k));
        if (!std::isfinite(vs1d(k))) {
            ++n_nonfinite;
        } else {
            vmin = std::min(vmin, vs1d(k));
            vmax = std::max(vmax, vs1d(k));
        }
    }
    logger.Error(fmt::format("  Vs min={:.4f}, max={:.4f} km/s over finite nodes, {} non-finite node(s); "
        "profile (z:Vs):{}", vmin, vmax, n_nonfinite, prof), MODULE_INV1D);

    for (int k = 0; k < vs1d.size(); ++k) {
        if (!std::isfinite(vs1d(k)) || vs1d(k) <= _0_CR) {
            logger.Error(fmt::format("  First non-physical Vs: node {} (z={} km) is {} after {} update(s).",
                k, zarr(k), vs1d(k), n_update), MODULE_INV1D);
            break;
        }
    }
}

void Inversion1D::check_pred_vel(const Eigen::VectorX<real_t>& pred_vel,
                                 const Eigen::VectorX<real_t>& periods,
                                 const Eigen::VectorX<real_t>& zarr,
                                 int iter, WaveType wt, SurfType tp) const {
    int first_bad = -1;
    for (int ip = 0; ip < pred_vel.size(); ++ip) {
        if (!std::isfinite(pred_vel(ip)) || pred_vel(ip) <= _0_CR) { first_bad = ip; break; }
    }
    if (first_bad < 0) return;

    auto& logger = ATTLogger::logger();
    const real_t bad = pred_vel(first_bad);
    if (bad == _0_CR) {
        // disper() writes exactly 0 from the first period whose root search fails
        logger.Error(fmt::format(
            "[{}] iter {}: dispersion solver found no fundamental-mode root from period {} s "
            "(index {} of {}); predicted velocities from here on are 0.",
            data_tag(wt, tp), iter, periods(first_bad), first_bad, periods.size()), MODULE_INV1D);
    } else {
        logger.Error(fmt::format(
            "[{}] iter {}: predicted velocity at period {} s (index {} of {}) is non-physical ({}).",
            data_tag(wt, tp), iter, periods(first_bad), first_bad, periods.size(), bad), MODULE_INV1D);
    }
    log_vs_profile(zarr, iter);
    abort_all_ranks();
}

void Inversion1D::check_kernels(const surfker::DepthKernel1D& kernels,
                                const Eigen::VectorX<real_t>& periods,
                                const Eigen::VectorX<real_t>& zarr,
                                int iter, WaveType wt, SurfType tp) const {
    // Empty matrices (sen_vp for Love waves) have no rows and pass trivially.
    int first_bad = -1;
    for (const auto* m : {&kernels.sen_vs, &kernels.sen_vp, &kernels.sen_rho}) {
        const int ib = first_nonfinite_row(*m);
        if (ib >= 0 && (first_bad < 0 || ib < first_bad)) first_bad = ib;
    }
    if (first_bad < 0) return;

    ATTLogger::logger().Error(fmt::format(
        "[{}] iter {}: depth kernel is non-finite at period {} s (index {} of {}){}",
        data_tag(wt, tp), iter, periods(first_bad), first_bad, periods.size(),
        tp == SurfType::GR
            ? "; group kernels re-run disper() at periods perturbed by +-0.5%, "
              "which can fail even when the nominal period succeeds."
            : "."), MODULE_INV1D);
    log_vs_profile(zarr, iter);
    abort_all_ranks();
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

    // Must precede anything that indexes zarr / vs1d (sigma, min/max below).
    check_inputs(zarr);

    real_t sigma = _0_CR;
    if (IP.postproc().smooth_method == 0) {
        sigma = IP.postproc().sigma[1];
    } else if (IP.postproc().smooth_method == 1) {
        sigma = 0.68 * (zarr(zarr.size() - 1) - zarr(0)) / IP.postproc().n_inv_grid[2];
    }
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

            Eigen::VectorX<real_t> pred_vel = surfker::surfdisp(req);
            check_pred_vel(pred_vel, sr.periods_info.periods, zarr, iter, wt, tp);
            real_t misfit = 0.5 * (pred_vel - sr.periods_info.meanvel).array().square().sum();
            logger.Debug(
                fmt::format("  iter {:3d} | {}_{} misfit={:.6e} (weight={:.3f})",
                    iter, waveTypeStr[static_cast<int>(wt)],
                    (itype == 0 ? "PH" : "GR"), misfit, IP.data().weights[itype]),
                MODULE_INV1D
            );
            misfit_total += misfit * IP.data().weights[itype];

            surfker::DepthKernel1D kernels = surfker::depthkernel1d(req);
            check_kernels(kernels, sr.periods_info.periods, zarr, iter, wt, tp);

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
        const int bad_node = first_nonfinite_row(update_total);
        if (bad_node >= 0) {
            // data, predictions and kernels are checked above, so this points
            // at the smoothing (e.g. sigma <= 0 gives exp(0/0) in gaussian_smooth_1d)
            logger.Error(fmt::format("iter {}: model update is non-finite at node {} (z={} km, value {}); "
                "check the smoothing sigma.", iter, bad_node, zarr(bad_node), update_total(bad_node)),
                MODULE_INV1D);
            abort_all_ranks();
        }
        const real_t update_norm = update_total.lpNorm<Eigen::Infinity>();
        if (update_norm > _0_CR) {
            update_total = step_length * update_total / update_norm;
        }  // zero update (perfect fit) would otherwise give 0/0 = NaN

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

    // The last update is applied after the final forward check, and the result
    // may seed the next wave type's inversion; make sure it is still physical.
    if (!vs1d.allFinite() || (vs1d.array() <= _0_CR).any()) {
        logger.Error("1D inversion produced a non-physical Vs model in its final update.", MODULE_INV1D);
        log_vs_profile(zarr, std::min(niter + 1, MAX_ITER_1D));
        abort_all_ranks();
    }

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
