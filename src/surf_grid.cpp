#include "surf_grid.h"

namespace {

inline int kernel_idx4(const int ix, const int iy, const int iz, const int iper,
                       const int ngrid_j, const int ngrid_k, const int nperiod_) {
    return (((ix * ngrid_j) + iy) * ngrid_k + iz) * nperiod_ + iper;
}

}

SurfGrid::SurfGrid(WaveType wt, SurfType vt){
    auto &sr = SrcRec::SR(wt, vt);
    nperiod_ = sr.periods_info.nperiod;
    Eigen::VectorX<real_t> periods = sr.periods_info.periods;
    wt_ = wt;
    itype_ = static_cast<int>(vt);
    type_name_ = waveTypeStr[static_cast<int>(wt)] + "_" + surfTypeStr[itype_];
    setup_active_kernels();

    auto& IP = InputParams::IP();
    auto& dcp = Decomposer::DCP();

    // Allocate shared memory for the grid arrays
    auto& mpi = Parallel::mpi();
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod_, a, win_a_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod_, b, win_b_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod_, c, win_c_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod_, topo_angle, win_topo_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod_, m11, win_m11_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod_, m12, win_m12_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod_, m22, win_m22_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod_, ref_t, win_ref_t_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod_, svel, win_svel_);

    // Initialize ref_t to 1.0
    if (mpi.is_node_main()) {
        std::fill(a, a + ngrid_i*ngrid_j*nperiod_, _0_CR);
        std::fill(b, b + ngrid_i*ngrid_j*nperiod_, _0_CR);
        std::fill(c, c + ngrid_i*ngrid_j*nperiod_, _0_CR);
        std::fill(topo_angle, topo_angle + ngrid_i*ngrid_j*nperiod_, _0_CR);
        std::fill(m11, m11 + ngrid_i*ngrid_j*nperiod_, _0_CR);
        std::fill(m12, m12 + ngrid_i*ngrid_j*nperiod_, _0_CR);
        std::fill(m22, m22 + ngrid_i*ngrid_j*nperiod_, _0_CR);
        std::fill(ref_t, ref_t + ngrid_i*ngrid_j*nperiod_, _1_CR);
    }
    mpi.barrier();

    if (run_mode == INVERSION_MODE || IP.inversion().model_para_type == MODEL_AZI_ANI) {
        if (run_mode == INVERSION_MODE) {
            for (int iper = 0; iper < nperiod_; ++iper) {
                adj_s_local.emplace_back(Eigen::MatrixX<real_t>::Zero(ngrid_i, ngrid_j));
                if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
                    adj_xi_local.emplace_back(Eigen::MatrixX<real_t>::Zero(ngrid_i, ngrid_j));
                    adj_eta_local.emplace_back(Eigen::MatrixX<real_t>::Zero(ngrid_i, ngrid_j));
                }
                if (IP.postproc().is_kden) {
                    kden_s_local.emplace_back(Eigen::MatrixX<real_t>::Zero(ngrid_i, ngrid_j));
                }
            }
        }

        sen_vp_loc = Tensor4r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k, nperiod_);
        sen_vs_loc = Tensor4r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k, nperiod_);
        sen_rho_loc = Tensor4r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k, nperiod_);
        sen_vp_loc.setZero();
        sen_vs_loc.setZero();
        sen_rho_loc.setZero();
        if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
            sen_gc_loc = Tensor4r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k, nperiod_);
            sen_gs_loc = Tensor4r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k, nperiod_);
            r1_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), nperiod_);
            r2_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), nperiod_);
            sen_gc_loc.setZero();
            sen_gs_loc.setZero();
            r1_loc.setZero();
            r2_loc.setZero();
        }
    }
    mpi.barrier();
}

void SurfGrid::setup_active_kernels() {
    auto& IP = InputParams::IP();
    active_kernels_ = std::vector<bool>(NPARAMS-1, false);
    active_kernels_[0] = true;  // vp
    if (IP.inversion().use_alpha_beta_rho && wt_ == WaveType::RL) {
        active_kernels_[1] = true;  // vs
        active_kernels_[2] = true;  // rho
    }
    if (IP.inversion().model_para_type == MODEL_AZI_ANI && wt_ == WaveType::RL) {
        active_kernels_[3] = true;  // gc
        active_kernels_[4] = true;  // gs
    }
}

void SurfGrid::build_media_matrix_with_topo() {
    auto& mpi = Parallel::mpi();
    auto& IP = InputParams::IP();
    auto &mg = ModelGrid::MG();
    auto &sr = SrcRec::SR(wt_, static_cast<SurfType>(itype_));
    const Eigen::VectorX<real_t>& periods = sr.periods_info.periods;

    // Load topography
    auto &topo = Topography::Topo();
    topo.grid(mg.xgrids, mg.ygrids);
    // Save the freshly-interpolated (unsmoothed) topo so each period can
    // start from the same base before period-specific smoothing.
    const Eigen::MatrixX<real_t> z_gridded = topo.z;

    Eigen::VectorX<real_t> avg_svel = Eigen::VectorX<real_t>::Zero(nperiod_);
    if (mpi.is_main()) {
        auto req = surfker::build_disp_req(mg.zgrids, mg.vs1d, periods,
                                IFLSPH, iwave_of(wt_), IMODE, itype_);

        avg_svel = surfker::surfdisp(req);
    }
    mpi.barrier();
    mpi.bcast(avg_svel.data(), nperiod_);

    const int n_elem = ngrid_i * ngrid_j * nperiod_;
    std::vector<real_t> tmp_angle = std::vector<real_t>(n_elem, _0_CR);
    std::vector<real_t> tmp_a = std::vector<real_t>(n_elem, _0_CR);
    std::vector<real_t> tmp_b = std::vector<real_t>(n_elem, _0_CR);
    std::vector<real_t> tmp_c = std::vector<real_t>(n_elem, _0_CR);
    for (int iper = 0; iper < nperiod_; ++iper){
        int loc_rank = mpi.select_rank_for_src(iper);
        if (mpi.rank() == loc_rank) {
            // Restore the unsmoothed gridded topo before each period's smoothing.
            topo.z = z_gridded;
            // sigma in km (wavelength-based), but gaussian_smooth_geo_2 expects arc-degrees.
            real_t sigma_km = avg_svel(iper) * periods(iper) *
                              IP.topo().wavelen_factor / (_2_CR * PI);
            real_t sigma_deg = sigma_km / (R_EARTH * DEG2RAD);
            topo.smooth(sigma_deg);

            // dip angle: shape (ngrid_i, ngrid_j)
            Eigen::MatrixX<real_t> angle_per = topo.calc_dip_angle();

            // geographic gradients fx (∂z/∂lon), fy (∂z/∂lat)
            Eigen::MatrixX<real_t> fx, fy;
            gradient_2_geo(topo.z, mg.xgrids, mg.ygrids, fx, fy);

            // media-matrix coefficients via Eigen array arithmetic (element-wise)
            auto fx2 = fx.array().square();
            auto fy2 = fy.array().square();
            auto denom = (1.0 + fx2 + fy2);   // (1 + fx² + fy²), shape (ni, nj)

            const Eigen::MatrixX<real_t> a_per = ((1.0 + fy2) / denom).matrix();
            const Eigen::MatrixX<real_t> b_per = ((1.0 + fx2) / denom).matrix();
            const Eigen::MatrixX<real_t> c_per = (fx.array() * fy.array() / denom).matrix();

            // Store as [ix][iy][iper], i.e. iper is the fastest-varying dimension.
            for (int ix = 0; ix < ngrid_i; ++ix) {
                for (int iy = 0; iy < ngrid_j; ++iy) {
                    const int idx = surf_idx(ix, iy, iper);
                    tmp_angle[idx] = angle_per(ix, iy);
                    tmp_a[idx] = a_per(ix, iy);
                    tmp_b[idx] = b_per(ix, iy);
                    tmp_c[idx] = c_per(ix, iy);
                }
            }
        }
    }

    // Same fix: reduce to private buffers first, then have node_main write to shared memory.
    mpi.sum_all_all_vect_inplace(tmp_angle);
    mpi.sum_all_all_vect_inplace(tmp_a);
    mpi.sum_all_all_vect_inplace(tmp_b);
    mpi.sum_all_all_vect_inplace(tmp_c);
    if (mpi.is_node_main()) {
        std::copy(tmp_angle.begin(), tmp_angle.end(), topo_angle);
        std::copy(tmp_a.begin(),     tmp_a.end(),     a);
        std::copy(tmp_b.begin(),     tmp_b.end(),     b);
        std::copy(tmp_c.begin(),     tmp_c.end(),     c);
    }
    mpi.barrier();
    mpi.sync_from_main_rank(topo_angle, n_elem);
    mpi.sync_from_main_rank(a, n_elem);
    mpi.sync_from_main_rank(b, n_elem);
    mpi.sync_from_main_rank(c, n_elem);
    mpi.barrier();
}

void SurfGrid::build_media() {
    auto &mpi = Parallel::mpi();
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();

    logger.Info(fmt::format("Building media matrix for {}...", type_name()), MODULE_GRID);
    int n_elem = ngrid_i * ngrid_j * nperiod_;
    if (IP.topo().is_consider_topo) {
        build_media_matrix_with_topo();
    } else {
        if (mpi.is_node_main()) {
            std::fill(a, a + n_elem, _1_CR);
            std::fill(b, b + n_elem, _1_CR);
            std::fill(c, c + n_elem, _0_CR);
        }
    }
    if (mpi.is_node_main()) {
        // Keep consistent with SurfATT-iso Fortran implementation:
        // m11 = a, m22 = b, m12 = -c
        std::copy(a, a + n_elem, m11);
        std::copy(b, b + n_elem, m22);
        std::transform(c, c + n_elem, m12, [](auto x) { return -x; });
    }
    mpi.barrier();
}

void SurfGrid::fwdsurf(){
    auto &mpi = Parallel::mpi();
    auto &logger = ATTLogger::logger();
    auto &IP = InputParams::IP();
    auto &mg = ModelGrid::MG();
    auto &dcp = Decomposer::DCP();
    auto &sr = SrcRec::SR(wt_, static_cast<SurfType>(itype_));
    const Eigen::VectorX<real_t>& periods = sr.periods_info.periods;

    logger.Info(fmt::format(
        "Computing {} velocity dispersion from 3d velocity model...", type_name()),
        MODULE_GRID
    );

    const int n_elem = ngrid_i * ngrid_j * nperiod_;
    std::vector<real_t> tmp_svel(n_elem, _0_CR);
    for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
        for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
            const int ix_glob = dcp.loc_I_start() + ix;
            const int iy_glob = dcp.loc_J_start() + iy;
            Eigen::VectorX<real_t> vs1d(ngrid_k);
            Eigen::VectorX<real_t> vp1d(ngrid_k);
            Eigen::VectorX<real_t> rho1d(ngrid_k);
            for (int k = 0; k < ngrid_k; ++k){
                if ( wt_ == WaveType::LV && IP.inversion().model_para_type == MODEL_RADIAL_ANI ) {
                    vs1d(k) = mg.vsh3d_loc(ix, iy, k);
                } else {
                    vs1d(k) = mg.vs3d_loc(ix, iy, k);
                }
                if ( IP.inversion().use_alpha_beta_rho ){
                    vp1d(k) = mg.vp3d_loc(ix, iy, k);
                    rho1d(k) = mg.rho3d_loc(ix, iy, k);
                } else {
                    vp1d(k) = vs2vp(vs1d(k));
                    rho1d(k) = vp2rho(vp1d(k));
                }
            }
            auto req = surfker::build_disp_req(mg.zgrids, vs1d, vp1d, rho1d, periods,
                                        IFLSPH, iwave_of(wt_), IMODE, itype_);
            Eigen::VectorX<real_t> svel_point = surfker::surfdisp(req);
            for (int iper = 0; iper < nperiod_; ++iper) {
                const int idx = surf_idx(ix_glob, iy_glob, iper);
                tmp_svel[idx] = svel_point(iper);
            }
        }
    }
    // Reduce into a private buffer first to avoid the shared-memory double-write
    // problem: svel is MPI shared memory; all node-local ranks point to the same
    // physical address, so MPI_Allreduce writing directly to svel would accumulate
    // the result nranks times instead of once.
    mpi.sum_all_all_vect_inplace(tmp_svel);
    if (mpi.is_node_main()) {
        std::copy(tmp_svel.begin(), tmp_svel.end(), svel);
    }
    mpi.barrier();
}

void SurfGrid::compute_dispersion_kernel() {
    auto& mpi = Parallel::mpi();
    auto& mg = ModelGrid::MG();
    auto& IP = InputParams::IP();
    auto& sr = SrcRec::SR(wt_, static_cast<SurfType>(itype_));
    auto& logger = ATTLogger::logger();
    auto& dcp = Decomposer::DCP();

    const Eigen::VectorX<real_t>& periods = sr.periods_info.periods;
    logger.Info(
        IP.inversion().model_para_type == MODEL_AZI_ANI ? "Computing anisotropic kernels on each surface grid point..."
                                     : "Computing isotropic kernels on each surface grid point...",
        MODULE_GRID
    );

    using MatRM = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
        for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
            Eigen::VectorX<real_t> vs1d(ngrid_k);
            Eigen::VectorX<real_t> vp1d(ngrid_k);
            Eigen::VectorX<real_t> rho1d(ngrid_k);
            for (int k = 0; k < ngrid_k; ++k){
                // For Love waves in radial anisotropy, dispersion depends on vsh
                if (wt_ == WaveType::LV && IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
                    vs1d(k) = mg.vsh3d_loc(ix, iy, k);
                } else {
                    vs1d(k) = mg.vs3d_loc(ix, iy, k);
                }
                if (IP.inversion().use_alpha_beta_rho) {
                    vp1d(k) = mg.vp3d_loc(ix, iy, k);
                    rho1d(k) = mg.rho3d_loc(ix, iy, k);
                } else {
                    vp1d(k) = vs2vp(vs1d(k));
                    rho1d(k) = vp2rho(vp1d(k));
                }
            }
            auto req = surfker::build_disp_req(mg.zgrids, vs1d, vp1d, rho1d, periods,
                                    IFLSPH, iwave_of(wt_), IMODE, itype_);
            surfker::DepthKernel1D kernels;
            if (IP.inversion().model_para_type != MODEL_AZI_ANI) {
                kernels = surfker::depthkernel1d(req);
            } else {
                kernels = surfker::depthkernelHTI1d(req);
            }
            
            // Copy the kernels for this grid point into the corresponding location in the global sensitivity arrays.
            const int id0 = kernel_idx4(ix, iy, 0, 0, dcp.loc_ny(), ngrid_k, nperiod_);
            Eigen::Map<MatRM> vs_block(sen_vs_loc.data() + id0, ngrid_k, nperiod_);
            vs_block = kernels.sen_vs.transpose();
            if (wt_ == WaveType::RL){
                Eigen::Map<MatRM> vp_block(sen_vp_loc.data() + id0, ngrid_k, nperiod_);
                Eigen::Map<MatRM> rho_block(sen_rho_loc.data() + id0, ngrid_k, nperiod_);
                vp_block = kernels.sen_vp.transpose();
                rho_block = kernels.sen_rho.transpose();
            }
            if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
                Eigen::Map<MatRM> gc_block(sen_gc_loc.data() + id0, ngrid_k, nperiod_);
                Eigen::Map<MatRM> gs_block(sen_gs_loc.data() + id0, ngrid_k, nperiod_);
                gc_block = kernels.sen_gc.transpose();
                gs_block = kernels.sen_gs.transpose();
            }
        }
    }
    mpi.barrier();
}

void SurfGrid::correct_depth_with_topo() {
    auto &dcp = Decomposer::DCP();
    auto &mg = ModelGrid::MG();
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();

    logger.Info("Correcting kernel depth with topography...", MODULE_GRID);

    // Tensor layout: (loc_nx, loc_ny, ngrid_k, nperiod), RowMajor
    // For fixed (ix, iy, iper), elements along iz are strided by nperiod.
    const Eigen::InnerStride<Eigen::Dynamic> iz_stride(nperiod_);
    using MapStride = Eigen::Map<Eigen::VectorX<real_t>, 0, Eigen::InnerStride<Eigen::Dynamic>>;

    const real_t zmin = mg.zgrids(0);
    const real_t zmax = mg.zgrids(ngrid_k - 1);

    for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
        for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
            int ix_glob = dcp.loc_I_start() + ix;
            int iy_glob = dcp.loc_J_start() + iy;

            for (int iper = 0; iper < nperiod_; ++iper) {
                // Topo-corrected depth grid: stretch each depth node by 1/cos(angle)
                real_t angle = topo_angle[surf_idx(ix_glob, iy_glob, iper)];
                real_t cosang = std::cos(angle * DEG2RAD);
                // Avoid numerical blow-up near vertical dip and keep interpolation in-bounds.
                if (!std::isfinite(cosang) || std::abs(cosang) < 1e-6) {
                    cosang = 1e-6;
                }
                Eigen::VectorX<real_t> newz = mg.zgrids / cosang;
                newz = newz.array().max(zmin).min(zmax).matrix();

                // Base offset in the tensor for this (ix, iy, iz=0, iper)
                int base = kernel_idx4(ix, iy, 0, iper, dcp.loc_ny(), ngrid_k, nperiod_);

                // Extract dense column (copy), interpolate, write back
                auto interp_kernel = [&](real_t* ptr) {
                    Eigen::VectorX<real_t> col(MapStride(ptr + base, ngrid_k, iz_stride));
                    Eigen::VectorX<real_t> col_new = interp1d(mg.zgrids, col, newz);
                    for (int iz = 0; iz < ngrid_k; ++iz)
                        ptr[base + iz * nperiod_] = col_new(iz);
                };

                interp_kernel(sen_vs_loc.data());
                interp_kernel(sen_vp_loc.data());
                interp_kernel(sen_rho_loc.data());
                if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
                    interp_kernel(sen_gc_loc.data());
                    interp_kernel(sen_gs_loc.data());
                }
            }
        }
    }
}

void SurfGrid::prepare_aniso_media() {
    auto& mpi  = Parallel::mpi();
    auto& mg   = ModelGrid::MG();
    auto& dcp  = Decomposer::DCP();
    auto& logger = ATTLogger::logger();

    logger.Info("Building anisotropic media for 2D eikonal solver...", MODULE_GRID);

    r1_loc.setZero();
    r2_loc.setZero();

    // Each rank sums over the depth dimension for its local (ix, iy) sub-domain.
    // sen_gc_loc shape: (loc_nx, loc_ny, ngrid_k, nperiod_)
    // gc3d_loc   shape: (loc_nx, loc_ny, ngrid_k)
    for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
        const int ix_glob = dcp.loc_I_start() + ix;
        for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
            const int iy_glob = dcp.loc_J_start() + iy;
            for (int iper = 0; iper < nperiod_; ++iper) {
                real_t r1_val = _0_CR, r2_val = _0_CR;
                for (int k = 0; k < ngrid_k; ++k) {
                    r1_val += sen_gc_loc(ix, iy, k, iper) * mg.gc3d_loc(ix, iy, k);
                    r2_val += sen_gs_loc(ix, iy, k, iper) * mg.gs3d_loc(ix, iy, k);
                }
                const real_t sv = svel[surf_idx(ix_glob, iy_glob, iper)];
                r1_loc(ix, iy, iper) = r1_val / sv;
                r2_loc(ix, iy, iper) = r2_val / sv;
            }
        }
    }

    // Scatter the local results into full-grid temporary buffers (others stay 0),
    // then allreduce so every rank holds the complete r1/r2 surface grids.
    const int n_elem = ngrid_i * ngrid_j * nperiod_;
    std::vector<real_t> tmp_r1(n_elem, _0_CR), tmp_r2(n_elem, _0_CR);
    for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
        const int ix_glob = dcp.loc_I_start() + ix;
        for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
            const int iy_glob = dcp.loc_J_start() + iy;
            for (int iper = 0; iper < nperiod_; ++iper) {
                const int idx = surf_idx(ix_glob, iy_glob, iper);
                tmp_r1[idx] = r1_loc(ix, iy, iper);
                tmp_r2[idx] = r2_loc(ix, iy, iper);
            }
        }
    }

    std::vector<real_t> full_r1(n_elem), full_r2(n_elem);
    mpi.sum_all_all(tmp_r1.data(), full_r1.data(), n_elem);
    mpi.sum_all_all(tmp_r2.data(), full_r2.data(), n_elem);

    // Compute updated media-matrix coefficients from r1/r2 and the existing
    // topography-corrected coefficients a, b, c (already in shared memory).
    // Formulas follow the SurfATT Fortran implementation:
    //   xi  = r1 / (1 + r1^2 + r2^2)
    //   eta = r2 / (1 + r1^2 + r2^2)
    //   m11 = a^2*(1+2*xi) - 4*a*c*eta + c^2*(1-2*xi) + (a+b-1)*(1-a)
    //   m12 = -a*c*(1+2*xi) + 2*c^2*eta + 2*a*b*eta - b*c*(1-2*xi) + (a+b-1)*c
    //   m22 = c^2*(1+2*xi) - 4*b*c*eta + b^2*(1-2*xi) + (a+b-1)*(1-b)
    if (mpi.is_node_main()) {
        for (int ix = 0; ix < ngrid_i; ++ix) {
            for (int iy = 0; iy < ngrid_j; ++iy) {
                for (int iper = 0; iper < nperiod_; ++iper) {
                    const int idx  = surf_idx(ix, iy, iper);
                    const real_t r1  = full_r1[idx];
                    const real_t r2  = full_r2[idx];
                    const real_t den = _1_CR + r1*r1 + r2*r2;
                    const real_t xi  = r1 / den;
                    const real_t eta = r2 / den;
                    const real_t a_  = a[idx];
                    const real_t b_  = b[idx];
                    const real_t c_  = c[idx];
                    m11[idx] = a_*a_*(_1_CR + _2_CR*xi) - _4_CR*a_*c_*eta
                                + c_*c_*(_1_CR - _2_CR*xi) + (a_ + b_ - _1_CR)*(_1_CR - a_);
                    m12[idx] = -a_*c_*(_1_CR + _2_CR*xi) + _2_CR*c_*c_*eta
                                + _2_CR*a_*b_*eta - b_*c_*(_1_CR - _2_CR*xi)
                                + (a_ + b_ - _1_CR)*c_;
                    m22[idx] = c_*c_*(_1_CR + _2_CR*xi) - _4_CR*b_*c_*eta
                                + b_*b_*(_1_CR - _2_CR*xi) + (a_ + b_ - _1_CR)*(_1_CR - b_);
                }
            }
        }
    }
    mpi.barrier();
}

void SurfGrid::release_shm() {
    auto& mpi = Parallel::mpi();
    mpi.free_shared(a, win_a_);
    mpi.free_shared(b, win_b_);
    mpi.free_shared(c, win_c_);
    mpi.free_shared(topo_angle, win_topo_);
    mpi.free_shared(m11, win_m11_);
    mpi.free_shared(m12, win_m12_);
    mpi.free_shared(m22, win_m22_);
    mpi.free_shared(ref_t, win_ref_t_);
    mpi.free_shared(svel, win_svel_);
}
