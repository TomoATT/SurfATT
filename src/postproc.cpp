#include "postproc.h"
#include "logger.h"
#include "decomposer.h"
#include "model_grid.h"
#include "utils.h"

namespace{
    // -----------------------------------------------------------------------
    // Spherical Laplacian for (longitude, latitude, depth) coordinates.
    //
    // The Laplace-Beltrami operator on a sphere in geographic coordinates
    // (φ = longitude, λ = latitude, z = depth with r = R_e − z) is:
    //
    //   ∇²u = (1/cos²λ) ∂²u/∂φ²            … longitude
    //        + ∂²u/∂λ²  − tan(λ) ∂u/∂λ      … latitude  (*)
    //        + ∂²u/∂z²  − [2/(R_e−z)] ∂u/∂z  … depth     (**)
    //
    // (*) The −tan(λ) term arises from the divergence of meridians on the
    //     sphere.  When angles are expressed in degrees an extra π/180
    //     factor appears (see derivation below).
    //
    // (**) The radial 2/r term is very small at seismological depths
    //      (≲0.03 %/km) but is included for correctness.
    //
    // Degree-space derivation for the tan term:
    //   Let Λ = latitude in degrees, λ = Λ·π/180.
    //   ∂/∂λ = (180/π) ∂/∂Λ  ⟹
    //   ∂²u/∂λ² − tan λ ∂u/∂λ
    //     = (180/π)² [∂²u/∂Λ² − (π/180) tan λ  ∂u/∂Λ]
    //
    //   The (180/π)² cancels with the same factor in ∂²u/∂Λ², so the
    //   finite-difference form per unit ch (deg²/step) is simply:
    //     ch * [ d²u/dΛ² − (π/180)·tan(λ)·du/dΛ ]
    // -----------------------------------------------------------------------
    Tensor3r compute_laplacian_3d_standard(const real_t ch, const real_t cv) {
        auto &dcp = Decomposer::DCP();
        const auto &mg = ModelGrid::MG();

        Tensor3r lap(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        lap.setZero();

        int ix_start, iy_start;
        int ib, ie, jb, je;

        // Determine boundary indices based on domain decomposition
        if (dcp.neighbors_id()[0] != -1) {
            ix_start = 1;
            ib = 0;
        } else {
            ix_start = 0;
            ib = 1;
        }
        if (dcp.neighbors_id()[1] != -1) {
            ie = dcp.loc_nx();
        } else {
            ie = dcp.loc_nx() - 1;
        }
        if (dcp.neighbors_id()[2] != -1) {
            iy_start = 1;
            jb = 0;
        } else {
            iy_start = 0;
            jb = 1;
        }
        if (dcp.neighbors_id()[3] != -1) {
            je = dcp.loc_ny();
        } else {
            je = dcp.loc_ny() - 1;
        }

        real_t dx2_inv = _1_CR / (dgrid_i * dgrid_i);
        real_t dy2_inv = _1_CR / (dgrid_j * dgrid_j);
        real_t dz2_inv = _1_CR / (dgrid_k * dgrid_k);
        real_t dy_inv_half = _1_CR / (_2_CR * dgrid_j);   // 1/(2·dΛ) for central diff
        real_t dz_inv_half = _1_CR / (_2_CR * dgrid_k);   // 1/(2·dz) for central diff

        // Compute spherical Laplacian
        for (int ix = ib; ix < ie; ++ix) {
            for (int iy = jb; iy < je; ++iy) {
                real_t lat_rad = dcp.y_loc_expd(iy+iy_start) * DEG2RAD;
                real_t cos_lat = std::cos(lat_rad);
                real_t dy2_corr_inv = _1_CR / (cos_lat * cos_lat);   // 1/cos²(λ)
                real_t tan_lat = std::tan(lat_rad);                   // tan(λ)

                for (int k = 0; k < ngrid_k; ++k) {
                    real_t center = dcp.expd_field(ix+ix_start, iy+iy_start, k);

                    // --- Longitude: ch / cos²(λ) · d²u/dφ² ---
                    real_t d2u_dx2 = ch * dx2_inv * dy2_corr_inv * (
                        dcp.expd_field(ix+ix_start+1, iy+iy_start, k) + 
                        dcp.expd_field(ix+ix_start-1, iy+iy_start, k) - 
                        _2_CR * center
                    );

                    // --- Latitude: ch · [d²u/dλ² − (π/180)·tan(λ)·du/dλ] ---
                    real_t u_jp1 = dcp.expd_field(ix+ix_start, iy+iy_start+1, k);
                    real_t u_jm1 = dcp.expd_field(ix+ix_start, iy+iy_start-1, k);

                    real_t d2u_dy2 = ch * dy2_inv * (u_jp1 + u_jm1 - _2_CR * center);
                    real_t du_dy   = ch * DEG2RAD * tan_lat * dy_inv_half * (u_jp1 - u_jm1);
                    // subtract the spherical correction (−tan·du/dλ → +tan·du/dΛ·π/180 changes sign)
                    real_t lat_term = d2u_dy2 - du_dy;

                    // --- Depth: cv · [d²u/dz² − 2/(R−z) · du/dz] ---
                    real_t u_kp1 = dcp.expd_field(ix+ix_start, iy+iy_start, k+1);
                    real_t u_km1 = dcp.expd_field(ix+ix_start, iy+iy_start, k-1);

                    real_t d2u_dz2 = cv * dz2_inv * (u_kp1 + u_km1 - _2_CR * center);
                    real_t r_inv = _2_CR / (R_EARTH - mg.zgrids(k));  // 2/(R_e − z)
                    real_t du_dz   = cv * r_inv * dz_inv_half * (u_kp1 - u_km1);
                    real_t depth_term = d2u_dz2 - du_dz;

                    lap(ix, iy, k) = d2u_dx2 + lat_term + depth_term;
                }
            }
        }
        return lap;
    }

    void apply_boundary_conditions(Tensor3r &arr) {
        auto &dcp = Decomposer::DCP();
        if (dcp.neighbors_id()[0] == -1) {
            for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
                for (int k = 0; k < ngrid_k; ++k) {
                    arr(0, iy, k) = arr(1, iy, k);  // Neumann BC at left boundary
                }
            }
        }
        if (dcp.neighbors_id()[1] == -1) {
            for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
                for (int k = 0; k < ngrid_k; ++k) {
                    arr(dcp.loc_nx() - 1, iy, k) = arr(dcp.loc_nx() - 2, iy, k);  // Neumann BC at right boundary
                }
            }
        }
        if (dcp.neighbors_id()[2] == -1) {
            for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
                for (int k = 0; k < ngrid_k; ++k) {
                    arr(ix, 0, k) = arr(ix, 1, k);  // Neumann BC at bottom boundary
                }
            }
        }
        if (dcp.neighbors_id()[3] == -1) {
            for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
                for (int k = 0; k < ngrid_k; ++k) {
                    arr(ix, dcp.loc_ny() - 1, k) = arr(ix, dcp.loc_ny() - 2, k);  // Neumann BC at top boundary
                }
            }
        }
        // Global boundaries in Z-direction
        for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
            for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
                arr(ix, iy, 0) = arr(ix, iy, 1);  // Neumann BC at bottom boundary in Z
                arr(ix, iy, ngrid_k - 1) = arr(ix, iy, ngrid_k - 2);  // Neumann BC at top boundary in Z
            }
        }
    }
}

void PostProc::InvGrid::init(const std::vector<int> &n_inv, int nset_) {
    const auto &mg = ModelGrid::MG();

    nset  = nset_;
    n_inv_I = n_inv[0];
    n_inv_J = n_inv[1];
    n_inv_K = n_inv[2];

    const int ninvx = n_inv_I + 2;
    const int ninvy = n_inv_J + 2;
    const int ninvz = n_inv_K + 2;

    const real_t x0 = mg.xgrids(0), x1 = mg.xgrids(mg.xgrids.size() - 1);
    const real_t y0 = mg.ygrids(0), y1 = mg.ygrids(mg.ygrids.size() - 1);
    const real_t z0 = mg.zgrids(0), z1 = mg.zgrids(mg.zgrids.size() - 1);

    const real_t dinvx = (x1 - x0) / n_inv_I;
    const real_t xadd  = dinvx / nset;
    const real_t x_beg = x0 - xadd, x_end = x1 + xadd;

    const real_t dinvy = (y1 - y0) / n_inv_J;
    const real_t yadd  = dinvy / nset;
    const real_t y_beg = y0 - yadd, y_end = y1 + yadd;

    const real_t dinvz = (z1 - z0) / n_inv_K;
    const real_t zadd  = dinvz / nset;
    const real_t z_beg = z0 - zadd, z_end = z1 + zadd;

    // inner 1-D reference nodes (without halos)
    Eigen::VectorX<real_t> x_inv_1d(n_inv_I), y_inv_1d(n_inv_J), z_inv_1d(n_inv_K);
    for (int i = 0; i < n_inv_I; ++i) x_inv_1d(i) = x0 + i * dinvx;
    for (int i = 0; i < n_inv_J; ++i) y_inv_1d(i) = y0 + i * dinvy;
    for (int i = 0; i < n_inv_K; ++i) z_inv_1d(i) = z0 + i * dinvz;

    xinv.resize(ninvx, nset);
    yinv.resize(ninvy, nset);
    zinv.resize(ninvz, nset);

    for (int is = 0; is < nset; ++is) {
        // fixed halo nodes
        xinv(0, is)        = x_beg;  xinv(ninvx - 1, is) = x_end;
        yinv(0, is)        = y_beg;  yinv(ninvy - 1, is) = y_end;
        zinv(0, is)        = z_beg;  zinv(ninvz - 1, is) = z_end;
        // inner nodes staggered by is * {x,y,z}add
        for (int j = 1; j < ninvx - 1; ++j) xinv(j, is) = x_inv_1d(j - 1) + is * xadd;
        for (int j = 1; j < ninvy - 1; ++j) yinv(j, is) = y_inv_1d(j - 1) + is * yadd;
        for (int j = 1; j < ninvz - 1; ++j) zinv(j, is) = z_inv_1d(j - 1) + is * zadd;
    }
}

PostProc::PostProc() {
    const auto &post = InputParams::IP().postproc();

    if ( post.smooth_method == 1) {
        inv_grid.init(post.n_inv_grid, post.n_inv_components);
        inv_grid_ani.init(post.n_inv_grid_ani, post.n_inv_components);
    }
}

std::vector<real_t> PostProc::InvGrid::fwd2inv(const Tensor3r &buf) {
    const auto &mg = ModelGrid::MG();
    const auto &dcp = Decomposer::DCP();
    auto &logger = ATTLogger::logger();
    auto &mpi = Parallel::mpi();

    int idx, idy, idz, m;
    int ninvx = n_inv_I + 2;
    int ninvy = n_inv_J + 2;
    int ninvz = n_inv_K + 2;
    int ngrid = ninvx * ninvy * ninvz * nset;
    real_t wx, wy, wz, wt;
    std::vector<real_t> tmp = std::vector<real_t>(ngrid, _0_CR);
    std::vector<real_t> weight = std::vector<real_t>(ngrid, _0_CR);
    std::vector<real_t> arr_inv = std::vector<real_t>(ngrid, _0_CR);

    for (int igrid = 0; igrid < nset; ++igrid) {
        for (int i = 0; i < dcp.loc_nx(); ++i) {
            idx = locate_bissection(xinv.col(igrid).data(), ninvx, mg.xgrids(i + dcp.loc_I_start()));
            if (idx == -1){
                logger.Error(fmt::format(
                    "x = {} out of bounds [{}, {}]", mg.xgrids(i + dcp.loc_I_start()), xinv(0, igrid), xinv(ninvx - 1, igrid)
                ), MODULE_POSTPROC);
                mpi.abort(EXIT_FAILURE);
            }
            wx = (mg.xgrids(i + dcp.loc_I_start()) - xinv(idx, igrid)) / (xinv(idx + 1, igrid) - xinv(idx, igrid));
            for (int j = 0; j < dcp.loc_ny(); ++j) {
                idy = locate_bissection(yinv.col(igrid).data(), ninvy, mg.ygrids(j + dcp.loc_J_start()));
                if (idy == -1){
                    logger.Error(fmt::format(
                        "y = {} out of bounds [{}, {}]", mg.ygrids(j + dcp.loc_J_start()), yinv(0, igrid), yinv(ninvy - 1, igrid)
                    ), MODULE_POSTPROC);
                    mpi.abort(EXIT_FAILURE);
                }
                wy = (mg.ygrids(j + dcp.loc_J_start()) - yinv(idy, igrid)) / (yinv(idy + 1, igrid) - yinv(idy, igrid));
                for (int k = 0; k < ngrid_k; ++k) {
                    idz = locate_bissection(zinv.col(igrid).data(), ninvz, mg.zgrids(k));
                    if (idz == -1){
                        logger.Error(fmt::format(
                            "z = {} out of bounds [{}, {}]", mg.zgrids(k), zinv(0, igrid), zinv(ninvz - 1, igrid)
                        ), MODULE_POSTPROC);
                        mpi.abort(EXIT_FAILURE);
                    }
                    wz = (mg.zgrids(k) - zinv(idz, igrid)) / (zinv(idz + 1, igrid) - zinv(idz, igrid));
                    for (int n = 0; n < 8; ++n) {
                        if (n == 0){
                            m = I2V_INV_GRIDS(idx, idy, idz, igrid);
                            wt = (_1_CR - wx) * (_1_CR - wy) * (_1_CR - wz);
                        } else if (n == 1) {
                            m = I2V_INV_GRIDS(idx, idy+1, idz, igrid);
                            wt = (_1_CR - wx) * wy * (_1_CR - wz);
                        } else if (n == 2) {
                            m = I2V_INV_GRIDS(idx+1, idy+1, idz, igrid);
                            wt = wx * wy * (_1_CR - wz);
                        } else if (n == 3) {
                            m = I2V_INV_GRIDS(idx+1, idy, idz, igrid);
                            wt = wx * (_1_CR - wy) * (_1_CR - wz);
                        } else if (n == 4) {
                            m = I2V_INV_GRIDS(idx, idy, idz+1, igrid);
                            wt = (_1_CR - wx) * (_1_CR - wy) * wz;
                        } else if (n == 5) {
                            m = I2V_INV_GRIDS(idx, idy+1, idz+1, igrid);
                            wt = (_1_CR - wx) * wy * wz;
                        } else if (n == 6) {
                            m = I2V_INV_GRIDS(idx+1, idy+1, idz+1, igrid);
                            wt = wx * wy * wz;
                        } else if (n == 7) {
                            m = I2V_INV_GRIDS(idx+1, idy, idz+1, igrid);
                            wt = wx * (_1_CR - wy)*wz;
                        }
                        weight[m] += wt;
                        tmp[m] += wt * buf(i, j, k);
                    }
                }
            }
        }
    }
    mpi.sum_all_all_vect_inplace(tmp);
    mpi.sum_all_all_vect_inplace(weight);
    for (int i = 0; i < ngrid; ++i) {
        if (weight[i] > _0_CR) {
            arr_inv[i] = tmp[i] / weight[i];
        } else {
            arr_inv[i] = _0_CR;
        }
    }
    return arr_inv;
}

Tensor3r PostProc::InvGrid::inv2fwd(const real_t *buf) {
    const auto &mg = ModelGrid::MG();
    const auto &dcp = Decomposer::DCP();
    auto &logger = ATTLogger::logger();
    auto &mpi = Parallel::mpi();

    int idx, idy, idz, m;
    int ninvx = n_inv_I + 2;
    int ninvy = n_inv_J + 2;
    int ninvz = n_inv_K + 2;
    real_t wx, wy, wz, wt;
    Tensor3r arr_fwd(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
    arr_fwd.setZero(); 

    for (int igrid = 0; igrid < nset; ++igrid) {
        for (int i = 0; i < dcp.loc_nx(); ++i) {
            idx = locate_bissection(xinv.col(igrid).data(), ninvx, mg.xgrids(i + dcp.loc_I_start()));
            if (idx == -1){
                logger.Error(fmt::format(
                    "x = {} out of bounds [{}, {}]", mg.xgrids(i + dcp.loc_I_start()), xinv(0, igrid), xinv(ninvx - 1, igrid)
                ), MODULE_POSTPROC);
                mpi.abort(EXIT_FAILURE);
            }
            wx = (mg.xgrids(i + dcp.loc_I_start()) - xinv(idx, igrid)) / (xinv(idx + 1, igrid) - xinv(idx, igrid));
            for (int j = 0; j < dcp.loc_ny(); ++j) {
                idy = locate_bissection(yinv.col(igrid).data(), ninvy, mg.ygrids(j + dcp.loc_J_start()));
                if (idy == -1){
                    logger.Error(fmt::format(
                        "y = {} out of bounds [{}, {}]", mg.ygrids(j + dcp.loc_J_start()), yinv(0, igrid), yinv(ninvy - 1, igrid)
                    ), MODULE_POSTPROC);
                    mpi.abort(EXIT_FAILURE);
                }
                wy = (mg.ygrids(j + dcp.loc_J_start()) - yinv(idy, igrid)) / (yinv(idy + 1, igrid) - yinv(idy, igrid));
                for (int k = 0; k < ngrid_k; ++k) {
                    idz = locate_bissection(zinv.col(igrid).data(), ninvz, mg.zgrids(k));
                    if (idz == -1){
                        logger.Error(fmt::format(
                            "z = {} out of bounds [{}, {}]", mg.zgrids(k), zinv(0, igrid), zinv(ninvz - 1, igrid)
                        ), MODULE_POSTPROC);
                        mpi.abort(EXIT_FAILURE);
                    }
                    wz = (mg.zgrids(k) - zinv(idz, igrid)) / (zinv(idz + 1, igrid) - zinv(idz, igrid));
                    real_t val = _0_CR;
                    for (int n = 0; n < 8; ++n) {
                        if (n == 0){
                            m = I2V_INV_GRIDS(idx, idy, idz, igrid);
                            wt = (_1_CR - wx) * (_1_CR - wy) * (_1_CR - wz);
                        } else if (n == 1) {
                            m = I2V_INV_GRIDS(idx, idy+1, idz, igrid);
                            wt = (_1_CR - wx) * wy * (_1_CR - wz);
                        } else if (n == 2) {
                            m = I2V_INV_GRIDS(idx+1, idy+1, idz, igrid);
                            wt = wx * wy * (_1_CR - wz);
                        } else if (n == 3) {
                            m = I2V_INV_GRIDS(idx+1, idy, idz, igrid);
                            wt = wx * (_1_CR - wy) * (_1_CR - wz);
                        } else if (n == 4) {
                            m = I2V_INV_GRIDS(idx, idy, idz+1, igrid);
                            wt = (_1_CR - wx) * (_1_CR - wy) * wz;
                        } else if (n == 5) {
                            m = I2V_INV_GRIDS(idx, idy+1, idz+1, igrid);
                            wt = (_1_CR - wx) * wy * wz;
                        } else if (n == 6) {
                            m = I2V_INV_GRIDS(idx+1, idy+1, idz+1, igrid);
                            wt = wx * wy * wz;
                        } else if (n == 7) {
                            m = I2V_INV_GRIDS(idx+1, idy, idz+1, igrid);
                            wt = wx * (_1_CR - wy)*wz;
                        }
                        val += wt * buf[m];
                    }
                    arr_fwd(i, j, k) += val;
                }
            }
        }
    }
    arr_fwd = arr_fwd / static_cast<real_t>(nset);  // average over staggered sets
    return arr_fwd;
}

Tensor3r PostProc::pde_smooth(const Tensor3r &buf, bool is_ani) {
    auto &logger = ATTLogger::logger();
    auto &dcp = Decomposer::DCP();
    auto &IP = InputParams::IP();
    auto &mpi = Parallel::mpi();

    // sigma_ani falls back to sigma when independent_smooth_ani is off (input_params)
    const auto &sigma = is_ani ? IP.postproc().sigma_ani : IP.postproc().sigma;
    real_t sigma_h = sigma[0];
    real_t sigma_v = sigma[1];

    if (sigma_h <= _0_CR || sigma_v <= _0_CR ) {
        logger.Error("Invalid sigma values for PDE-based smoothing. All sigma values must be positive.", MODULE_POSTPROC);
        mpi.abort(EXIT_FAILURE);
    }
    Tensor3r smoothed_buf(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
    smoothed_buf = buf;  // initialize smoothed_buf with the input buf

    // Compute per-direction maximal stable diffusion coefficients (dt = 1).
    // Horizontal uses degrees², vertical uses km² — kept separate so units never mix.
    // Stability condition for 3-D forward Euler with spherical metric:
    //   ch/(dx²·cos²λ) + ch/dy² + cv/dz² ≤ 1/2
    // The 1/cos²λ factor on the longitude term is largest at the highest |latitude|
    // in the domain, so we must use cos²(lat_max) in the stability bound.
    const auto &mg = ModelGrid::MG();
    real_t lat_max = std::max(std::abs(mg.ygrids(0)),
                              std::abs(mg.ygrids(ngrid_j - 1)));
    // Clamp to avoid cos²→0 at the poles (cap at 89°)
    lat_max = std::min(lat_max, static_cast<real_t>(89.0));
    real_t cos2_lat_max = std::pow(std::cos(lat_max * DEG2RAD), 2);

    // dx_eff² = dx² · cos²(lat_max):  effective longitude spacing at worst latitude
    real_t cmax_h = std::min(dgrid_i * dgrid_i * cos2_lat_max, dgrid_j * dgrid_j) / 6.0;  // deg²
    real_t cmax_v = dgrid_k * dgrid_k / 6.0;  // km²
    if (cmax_h <= VERYTINY || cmax_v <= VERYTINY) {
        logger.Error("Grid spacing is too small for PDE-based smoothing.", MODULE_POSTPROC);
        mpi.abort(EXIT_FAILURE);
    }

    // Time step (dimensionless); stability is enforced via cmax_h/cmax_v below.
    const real_t dt = _1_CR;

    // Minimum steps required for each direction to achieve its target sigma.
    // sigma_h is in degrees, sigma_v is in km — each compared only against
    // its own-unit cmax, so there is no cross-unit comparison.
    int nstep_h = static_cast<int>(std::ceil(sigma_h * sigma_h / (2.0 * cmax_h * dt)));
    int nstep_v = static_cast<int>(std::ceil(sigma_v * sigma_v / (2.0 * cmax_v * dt)));
    int nstep   = std::max({nstep_h, nstep_v, 1});

    // Actual diffusion coefficients that achieve exactly sigma in nstep steps.
    // Because nstep ≥ nstep_h and nstep ≥ nstep_v, both ch ≤ cmax_h and cv ≤ cmax_v,
    // so the stability condition is guaranteed.
    real_t ch = sigma_h * sigma_h / (2.0 * nstep * dt);  // deg²/step
    real_t cv = sigma_v * sigma_v / (2.0 * nstep * dt);  // km²/step

    logger.Debug(fmt::format(
        "PDE smooth: sigma_h={:.3f} deg, sigma_v={:.3f} km, nstep={} (h:{} v:{}), ch={:.4e}, cv={:.4e}",
        sigma_h, sigma_v, nstep, nstep_h, nstep_v, ch, cv), MODULE_POSTPROC);

    mpi.barrier();

    for (int istep = 0; istep < nstep; ++istep) {
        // Prepare ghost cells for domain decomposition
        dcp.prepare_expanded_field(smoothed_buf.data());

        // Compute Laplacian using selected stencil
        Tensor3r lap = compute_laplacian_3d_standard(ch, cv);

        // update smoothed_buf
        smoothed_buf = smoothed_buf + dt * lap;

        // Apply boundary conditions
        apply_boundary_conditions(smoothed_buf);

        mpi.barrier();
    }
    return smoothed_buf;
}

Tensor3r PostProc::smooth(const Tensor3r &buf, bool is_ani) {
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();
    auto &mpi = Parallel::mpi();

    if (IP.postproc().smooth_method == 0) {
        return pde_smooth(buf, is_ani);
    } else if (IP.postproc().smooth_method == 1) {
        InvGrid &ig = is_ani ? inv_grid_ani : inv_grid;
        std::vector<real_t> inv_buf = ig.fwd2inv(buf);
        // Here we can apply some smoothing in the inversion grid if needed (not implemented in this example)
        return ig.inv2fwd(inv_buf.data());
    } else {
        logger.Error("Invalid smoothing method specified in input parameters.", MODULE_POSTPROC);
        mpi.abort(EXIT_FAILURE);
    }
}


// -------------------------------------------------------------------------------
// ------- namespace postproc functions that are called from inversion.cpp -------
// -------------------------------------------------------------------------------

void postproc::kernel_precondition(SurfGrid& sg) {
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();
    auto &mpi = Parallel::mpi();

    if ( !IP.postproc().is_kden ) return;

    logger.Info("Preconditioning kernels with reference model parameters...", MODULE_POSTPROC);
    // normalize kernel density by L_inf
    Eigen::Tensor<real_t, 0, Eigen::RowMajor> L_inf_tensor = sg.ker_den_loc.abs().maximum();
    real_t L_inf = L_inf_tensor();
    // Reduce to the global max before the threshold test: testing the rank-local
    // max would let uncovered ranks return early and mismatch the collectives.
    mpi.barrier();
    mpi.max_all_all_inplace(L_inf);
    if (L_inf < VERYTINY) {
        logger.Warn("Kernel density is too small, skipping preconditioning.", MODULE_POSTPROC);
        return;
    }

    // Match legacy Fortran: precond = |density|/max, zeroed below threshold,
    // otherwise 1/precond^kdensity_coe
    const real_t precond_thres = static_cast<real_t>(1e-4);
    const real_t coe = IP.postproc().kdensity_coe;
    Tensor3r ken_den_norm = sg.ker_den_loc.abs() / L_inf;
    Tensor3r hess_inv = (ken_den_norm >= precond_thres).select(
            ken_den_norm.pow(-coe),
            ken_den_norm.constant(_0_CR)
        );

    // Precondition the kernels by multiplying with the reference model parameters at each surface grid point
    int nker = NPARAMS - 1;
    for (int iparam = 0; iparam < nker; ++iparam) {
        if (sg.is_active_ker(iparam)) {
            sg.ker_loc[iparam] = sg.ker_loc[iparam] * hess_inv;
        }
    }
}

FieldVec postproc::kernel_smooth(const SurfGrid& sg) {
    auto &logger = ATTLogger::logger();
    auto &PP = PostProc::PP();
    auto &IP = InputParams::IP();

    std::string method_name = (IP.postproc().smooth_method == 0) ? "PDE smoothing" : "multigrid smoothing";

    logger.Info(fmt::format("Regularizing kernels with {}...", method_name), MODULE_POSTPROC);
    FieldVec ker_loc_smooth(NPARAMS);
    for (int iparam = 0; iparam < NPARAMS - 1; ++iparam) {
        if (sg.is_active_ker(iparam)) {
            const bool is_ani_param = (iparam == 3 || iparam == 4);  // gc, gs
            ker_loc_smooth[iparam] = PP.smooth(sg.ker_loc[iparam], is_ani_param);
        }
    }
    if (sg.wave_type() == WaveType::LV)
        ker_loc_smooth[NPARAMS - 1] = ker_loc_smooth[0];
    return ker_loc_smooth;
}

