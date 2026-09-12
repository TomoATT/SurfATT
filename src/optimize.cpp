#include "optimize.h"
#include "config.h"
#include "decomposer.h"
#include "parallel.h"
#include "h5io.h"
#include "logger.h"
#include "input_params.h"
#include "model_grid.h"

#include <algorithm>
#include <cmath>

namespace optimize {

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

// Element-wise dot product summed over all active parameters.
static real_t field_dot(const FieldVec &a, const FieldVec &b) {
    real_t result = _0_CR;
    for (size_t p = 0; p < a.size(); ++p) {
        if (a[p].size() > 0 && b[p].size() > 0) {
            Eigen::Tensor<real_t, 0, Eigen::RowMajor> d = (a[p] * b[p]).sum();
            result += d();
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// calc_descent_angle
// ---------------------------------------------------------------------------
real_t calc_descent_angle(const FieldVec &direction, const FieldVec &gradient) {
    auto &mpi = Parallel::mpi();

    // Compute local contributions.
    // Angle between actual step (–direction) and steepest descent (–gradient):
    // cos = (direction·gradient) / (||direction|| * ||gradient||)
    real_t local_dot   = field_dot(gradient, direction);
    real_t local_gnorm =  field_dot(gradient, gradient);
    real_t local_dnorm =  field_dot(direction, direction);

    // Reduce across all MPI ranks.
    real_t global_dot, global_gnorm, global_dnorm;
    mpi.sum_all_all(local_dot,   global_dot);
    mpi.sum_all_all(local_gnorm, global_gnorm);
    mpi.sum_all_all(local_dnorm, global_dnorm);
    global_gnorm = std::sqrt(global_gnorm);
    global_dnorm = std::sqrt(global_dnorm);

    // Guard against zero norms.
    if (global_gnorm < REAL_EPS || global_dnorm < REAL_EPS)
        return static_cast<real_t>(90.0);

    real_t cos_angle = global_dot / (global_gnorm * global_dnorm);
    // Clamp to [-1, 1] to avoid NaN from acos due to floating-point errors.
    cos_angle = std::max(_M_1_CR, std::min(_1_CR, cos_angle));
    return std::acos(cos_angle) * RAD2DEG;
}

// ---------------------------------------------------------------------------
// field_dot_global  — MPI-reduced directional derivative magnitude
// ---------------------------------------------------------------------------
// The adjoint kernels are densities over the horizontal surface, while their
// depth dimension already consists of discrete layer sensitivities.  This
// routine therefore adds only the horizontal quadrature weight.
//
// No per-parameter chain rule from alpha to the physical model is needed:
// combine_kernels() already returns dchi/dln(m) for the multiplicatively
// updated parameters (vs/vp/rho/gamma) and dchi/dm for the additive ones
// (gc/gs), so in both cases the step is x(alpha) = x0 - alpha*d and
// -dx/dalpha = d.
//
// dlon_i and dlat_j are nodal quadrature widths.  End points receive half of
// the adjacent interval, as in the trapezoidal rule.  No dz factor is added.
static real_t nodal_width_rad(const Eigen::VectorX<real_t> &coords_deg,
                              int index) {
    const int n = static_cast<int>(coords_deg.size());
    if (n < 2) return _0_CR;

    real_t width_deg;
    if (index == 0) {
        width_deg = std::abs(coords_deg(1) - coords_deg(0)) * _0_5_CR;
    } else if (index == n - 1) {
        width_deg = std::abs(coords_deg(n - 1) - coords_deg(n - 2)) * _0_5_CR;
    } else {
        width_deg = std::abs(coords_deg(index + 1) - coords_deg(index - 1))
                    * _0_5_CR;
    }
    return width_deg * DEG2RAD;
}

static real_t field_dot_global(const FieldVec &kernel,
                               const FieldVec &direction) {
    auto &dcp = Decomposer::DCP();
    auto &mg  = ModelGrid::MG();
    auto &mpi = Parallel::mpi();
    auto &IP  = InputParams::IP();

    real_t local = _0_CR;
    for (size_t p = 0; p < kernel.size(); ++p) {
        if (kernel[p].size() == 0 || direction[p].size() == 0) continue;

        for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
            const int ix_global = dcp.loc_I_start() + ix;
            const real_t dlon = nodal_width_rad(mg.xgrids, ix_global);

            for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
                const int iy_global = dcp.loc_J_start() + iy;
                const real_t dlat = nodal_width_rad(mg.ygrids, iy_global);
                const real_t latitude = mg.ygrids(iy_global) * DEG2RAD;
                const real_t area =
                    R_EARTH * R_EARTH * std::abs(std::cos(latitude))
                    * dlon * dlat;

                for (int iz = 0; iz < ngrid_k; ++iz) {
                    const int index = I2V(ix_global, iy_global, iz);
                    real_t minus_dm_dalpha;
                    real_t kernel_val = kernel[p](ix, iy, iz);

                    if (p == 0) {
                        // vs(alpha) = vs0 * (1 - alpha*d_vs)
                        minus_dm_dalpha =
                            mg.vs3d[index] * direction[p](ix, iy, iz);
                        if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
                            kernel_val = kernel[0](ix, iy, iz) + kernel[5](ix, iy, iz);
                        }
                    } else if (p == 1) {
                        // vp(alpha) = vp0 * (1 - alpha*d_vp)
                        minus_dm_dalpha =
                            mg.vp3d[index] * direction[p](ix, iy, iz);
                    } else if (p == 2) {
                        // rho(alpha) = rho0 * (1 - alpha*d_rho)
                        minus_dm_dalpha =
                            mg.rho3d[index] * direction[p](ix, iy, iz);
                    } else if (p == 5 &&
                               IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
                        minus_dm_dalpha = (mg.vsh3d[index] / mg.vs3d[index]) * direction[p](ix, iy, iz);
                    } else {
                        // gc and gs are updated additively.
                        minus_dm_dalpha = direction[p](ix, iy, iz);
                    }

                    local += kernel_val
                             * minus_dm_dalpha * area;
                }
            }
        }
    }

    real_t global;
    mpi.sum_all_all(local, global);
    return global;
}

// ---------------------------------------------------------------------------
// wolfe_condition
// ---------------------------------------------------------------------------
WolfeResult wolfe_condition(const FieldVec &gradient, const FieldVec &ker_next,
                            const FieldVec &direction,
                            real_t alpha, real_t &alpha_L, real_t &alpha_R,
                            real_t f0, real_t f1,
                            int subiter) {
    auto &logger = ATTLogger::logger();
    auto &IP     = InputParams::IP();
    const real_t c1            = IP.inversion().c1;
    const real_t c2            = IP.inversion().c2;
    const int    max_sub_niter = IP.inversion().max_sub_niter;
    const real_t alpha_init    = IP.inversion().step_length;

    // direction is the positive gradient (search_dir); model_update subtracts it,
    // so the actual descent step is d = -direction.  Wolfe conditions require
    // q = ∇f · d < 0, hence we negate the dot products here.
    // q is evaluated at the base model (alpha=0).  q1 is evaluated at the
    // current trial point.  For the linear multiplicative updates their tangent
    // is constant; path_alpha matters only for vsh = vs*gamma.
    const real_t q  = -field_dot_global(gradient, direction);
    const real_t q1 = -field_dot_global(ker_next,  direction);

    const bool cond_armijo    = f1 <= f0 + alpha * c1 * q;
    const bool cond_curvature = q1 >= c2 * q;

    logger.Debug(fmt::format("q = {:.6e}, q1 = {:.6e}, f0 = {:.6e}, f1 = {:.6e}, alpha = {:.6e}, c1*q = {:.6e}, c2*q = {:.6e}",
        q, q1, f0, f1, alpha, c1 * q, c2 * q), MODULE_OPTIM);
    logger.Debug(fmt::format("Armijo condition: f0={:.6e}  f1={:.6e}  f0+c1*alpha*q={:.6e}",
        f0, f1, f0 + alpha * c1 * q), MODULE_OPTIM);
    logger.Debug(fmt::format("Curvature condition: q(grad·dir)={:.6e}  q1(grad'·dir)={:.6e}  c2*q={:.6e}",
        q, q1, c2 * q), MODULE_OPTIM);
    logger.Info(fmt::format("Armijo: {};  Curvature: {}", cond_armijo, cond_curvature), MODULE_OPTIM);

    WolfeResult res;

    if (cond_armijo && cond_curvature) {
        res.status     = WolfeResult::Status::ACCEPT;
        res.next_alpha = alpha;
        logger.Info(fmt::format("Wolfe conditions satisfied. Misfit {:.6f} -> {:.6f}",
            f0, f1, alpha), MODULE_OPTIM);

    } else if (cond_armijo && alpha_R > _0_CR) {
        // Armijo previously failed, so its upper bound has priority. Once a
        // shorter trial satisfies Armijo, accept it even if curvature does not.
        res.status     = WolfeResult::Status::ACCEPT;
        res.next_alpha = alpha;
        logger.Info(fmt::format(
            "Armijo satisfied after shrinking the step. Accept alpha={:.6f} without further curvature trials.",
            alpha), MODULE_OPTIM);

    } else if (!cond_armijo && !cond_curvature) {
        // Both conditions failed: prioritize Armijo and shrink directly.
        alpha_R        = alpha;
        res.next_alpha = alpha * _0_5_CR;
        res.status     = WolfeResult::Status::TRY;
        logger.Info(fmt::format(
            "Neither Armijo nor curvature satisfied; prioritize Armijo. Try alpha={:.6f}.",
            res.next_alpha), MODULE_OPTIM);

    } else if (!cond_armijo) {
        // Step too large: shrink upper bracket.
        alpha_R        = alpha;
        res.next_alpha = (alpha_L + alpha_R) * _0_5_CR;
        res.status     = WolfeResult::Status::TRY;
        logger.Info(fmt::format("Armijo not satisfied (step too large). Try alpha={:.6f}.",
            res.next_alpha), MODULE_OPTIM);

    } else {
        // Armijo ok but curvature not: step too small, widen.
        alpha_L = alpha;
        real_t candidate = (alpha_R > _0_CR) ? (alpha_L + alpha_R) * _0_5_CR
                                             : alpha * _2_CR;
        
        // Cap at initial configured step length; if capped, accept even if curvature not satisfied
        if (candidate > alpha_init) {
            res.next_alpha = alpha_init;
            res.status     = WolfeResult::Status::ACCEPT;
            logger.Info(fmt::format("Curvature not satisfied but enlarged alpha exceeds alpha_init={:.6f}. ",
                alpha_init, alpha_init), MODULE_OPTIM);
        } else {
            res.next_alpha = candidate;
            res.status     = WolfeResult::Status::TRY;
            logger.Info(fmt::format("Curvature not satisfied (step too small)",
                res.next_alpha), MODULE_OPTIM);
        }
    }

    if (res.status == WolfeResult::Status::TRY && subiter == max_sub_niter - 1) {
        res.status     = WolfeResult::Status::FAIL;
        res.next_alpha = alpha;
        logger.Info("Wolfe condition: maximum sub-iterations reached without convergence.", MODULE_OPTIM);
    }

    return res;
}

// ---------------------------------------------------------------------------
// lbfgs_direction
// ---------------------------------------------------------------------------
FieldVec lbfgs_direction(int iter) {
    auto &dcp = Decomposer::DCP();
    auto &mpi = Parallel::mpi();

    const int hist_size  = std::min(iter, MAX_LBFGS_STORE); // number of (s, y) pairs available
    const int hist_start = iter - hist_size;  // index of the oldest stored pair

    // Dataset name suffix for a given iteration k: "_NNN"
    auto sfx = [](int k) { return fmt::format("_{:03d}", k); };

    // Parameter short names matching store_model / store_gradient

    // Allocate global-size direction tensors on every rank.
    // Content is computed only on main rank; other ranks keep zeros
    // (distribute_data reads only from main rank's buffer).
    FieldVec dir_global(NPARAMS);
    for (int p = 0; p < NPARAMS; ++p) {
        dir_global[p] = Tensor3r(ngrid_i, ngrid_j, ngrid_k);
        dir_global[p].setZero();
    }

    if (mpi.is_main()) {
        H5IO f(db_fname, H5IO::RDONLY);

        // ---- q ← current gradient (read from HDF5) -------------------------
        FieldVec q(NPARAMS);
        for (int p = 0; p < NPARAMS; ++p) {
            if (is_active_param[p])
                q[p] = f.read_tensor<real_t>(std::string("grad_") + pnames[p] + sfx(iter));
        }

        // ---- Load history pairs (s_h, y_h) for h in [0, hist_size) ---------
        // s_h = model_{h+1} - model_{h}
        // y_h = grad_{h+1}  - grad_{h}
        std::vector<FieldVec> s_hist(hist_size, FieldVec(NPARAMS));
        std::vector<FieldVec> y_hist(hist_size, FieldVec(NPARAMS));
        std::vector<real_t>   rho(hist_size, _0_CR);

        for (int h = 0; h < hist_size; ++h) {
            int ik = hist_start + h;
            real_t sy = _0_CR;
            for (int p = 0; p < NPARAMS; ++p) {
                if (is_active_param[p]) {
                    auto mk  = f.read_tensor<real_t>(std::string("model_") + pnames[p] + sfx(ik));
                    auto mk1 = f.read_tensor<real_t>(std::string("model_") + pnames[p] + sfx(ik + 1));
                    // vs/vp/rho (p < 3) are updated multiplicatively, so work in log space;
                    // gc/gs (p >= 3) are updated additively, use the raw difference.
                    if (p < 3 || p == 5) // vs, vp, rho, gamma
                        s_hist[h][p] = mk1.log() - mk.log();
                    else
                        s_hist[h][p] = mk1 - mk;

                    auto gk  = f.read_tensor<real_t>(std::string("grad_") + pnames[p] + sfx(ik));
                    auto gk1 = f.read_tensor<real_t>(std::string("grad_") + pnames[p] + sfx(ik + 1));
                    y_hist[h][p] = gk1 - gk;

                    Eigen::Tensor<real_t, 0, Eigen::RowMajor> d = (s_hist[h][p] * y_hist[h][p]).sum();
                    sy += d();
                }
            }
            rho[h] = (sy > _0_CR) ? (real_t{1} / sy) : _0_CR;
        }

        // ---- First loop (backward: newest → oldest) -------------------------
        // Skip pairs with rho = 0 (sy ≤ 0: curvature condition violated).
        std::vector<real_t> alpha(hist_size, _0_CR);
        for (int h = hist_size - 1; h >= 0; --h) {
            alpha[h] = rho[h] * field_dot(s_hist[h], q);
            for (int p = 0; p < NPARAMS; ++p)
                if (is_active_param[p])
                    q[p] = q[p] - alpha[h] * y_hist[h][p];
        }

        // ---- Initial Hessian scaling: γ = (s^T y) / (y^T y) ---------------
        // Clamp γ to positive: a negative or zero γ would invert or zero the
        // search direction (causing the 90-degree restart loop).
        auto r = q; // r ← H₀ q
        if (hist_size > 0) {
            int last  = hist_size - 1;
            real_t yy = field_dot(y_hist[last], y_hist[last]);
            real_t sy = field_dot(s_hist[last], y_hist[last]);
            real_t gamma = (sy > _0_CR && yy > _0_CR) ? (sy / yy) : _1_CR;
            for (int p = 0; p < NPARAMS; ++p)
                if (is_active_param[p])
                    r[p] = gamma * r[p];
        }

        // ---- Second loop (forward: oldest → newest) -------------------------
        for (int h = 0; h < hist_size; ++h) {
            real_t beta = rho[h] * field_dot(y_hist[h], r);
            for (int p = 0; p < NPARAMS; ++p)
                if (is_active_param[p])
                    r[p] = r[p] + (alpha[h] - beta) * s_hist[h][p];
        }

        // ---- Search direction = -H_k g_k ------------------------------------
        for (int p = 0; p < NPARAMS; ++p)
            if (is_active_param[p])
                dir_global[p] = r[p];
    }

    // Distribute the global direction from main rank to all ranks.
    FieldVec dir_loc(NPARAMS);
    for (int p = 0; p < NPARAMS; ++p)
        if (is_active_param[p])
            dir_loc[p] = dcp.distribute_data(dir_global[p].data());

    return dir_loc;
}

} // namespace optimize
