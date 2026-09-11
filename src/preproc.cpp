#include "preproc.h"
#include "eikonal_solver.h"
#include "utils.h"
#include "decomposer.h"
#include "config.h"

namespace{
Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
extract_period_ij(const real_t* buf, int np, int iper) {
    using MatRM = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Strd  = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    using MapMatConst = Eigen::Map<const MatRM, 0, Strd>;
    // Each period's data is stored contiguously in memory, so we can create a Map with the appropriate stride to access it.
    const Strd stride(ngrid_j * np, np);
    // Map the buffer to a matrix view for the specified period (iper).
    MapMatConst view(buf + iper, ngrid_i, ngrid_j, stride);
    return view;
}

void accumulate_kernels(
    SurfGrid& sg, const int iper,
    const Eigen::MatrixX<real_t>& adj_field,
    const Eigen::MatrixX<real_t>* kden_field = nullptr,
    const Eigen::MatrixX<real_t>* tfield = nullptr
) {
    auto& IP = InputParams::IP();
    auto& mg = ModelGrid::MG();

    auto svel_ij = extract_period_ij(sg.svel, sg.nperiod(), iper);
    const auto inv_svel3 = svel_ij.array().pow(-_3_CR);

    sg.adj_s_local[iper].array() += adj_field.array() * inv_svel3;
    if (kden_field != nullptr ) {
        sg.kden_s_local[iper].array() += kden_field->array() * inv_svel3;
    }

    // If anisotropy is considered and the travel time field (tfield) is available, compute the adjoint sources for the anisotropy parameters xi and eta.
    if (IP.inversion().model_para_type == MODEL_AZI_ANI && tfield != nullptr) {
        Eigen::MatrixX<real_t> Tx, Ty;
        gradient_2_geo(*tfield, mg.xgrids, mg.ygrids, Tx, Ty);

        auto a_ij = extract_period_ij(sg.a, sg.nperiod(), iper);
        auto b_ij = extract_period_ij(sg.b, sg.nperiod(), iper);
        auto c_ij = extract_period_ij(sg.c, sg.nperiod(), iper);

        auto Tx2 = Tx.array().square();
        auto Ty2 = Ty.array().square();
        auto TxTy = (Tx.array() * Ty.array());
        auto a2 = a_ij.array().square();
        auto b2 = b_ij.array().square();
        auto c2 = c_ij.array().square();
        auto ac = (a_ij.array() * c_ij.array());
        auto bc = (b_ij.array() * c_ij.array());
        auto ab = (a_ij.array() * b_ij.array());

        // adj_xi: adjtable * (Tx*(Tx*(-a²+c²) + Ty*(ac - bc))
        //                   + Ty*(Tx*(ac - bc) + Ty*(-c²+b²)))
        sg.adj_xi_local[iper].array() += adj_field.array() * (
            Tx2 * (-a2 + c2) + _2_CR * TxTy * (ac - bc) + Ty2 * (-c2 + b2)
        );

        // adj_eta: adjtable * (Tx*(2*ac*Tx - (ab+c²)*Ty)
        //                    + Ty*(-(ab+c²)*Tx + 2*bc*Ty))
        sg.adj_eta_local[iper].array() += adj_field.array() * (
            _2_CR * ac * Tx2 - _2_CR * (ab + c2) * TxTy + _2_CR * bc * Ty2
        );
    }
}

} // namespace

real_t preproc::forward_for_event(SrcRec& sr, SurfGrid& sg, const bool is_calc_adj) {
    auto& IP = InputParams::IP();
    auto& mg = ModelGrid::MG();
    auto& mpi = Parallel::mpi();
    auto& logger = ATTLogger::logger();

    real_t chi = _0_CR;
    logger.Info("Computing forward or/and adjoint fields for each event...", MODULE_PREPROC);
    for (auto& evt : sr.events_local) {
        real_t evla = evt.second.evla;
        real_t evlo = evt.second.evlo;
        int iper = evt.second.iper;

        logger.Debug(fmt::format("Rank:{}, Computing travel time for event: {} ",
            mpi.rank(), evt.first), 
            MODULE_PREPROC, false
        );
        auto svel_ij = extract_period_ij(sg.svel, sg.nperiod(), iper);
        auto m11_ij = extract_period_ij(sg.m11, sg.nperiod(), iper);
        auto m12_ij = extract_period_ij(sg.m12, sg.nperiod(), iper);
        auto m22_ij = extract_period_ij(sg.m22, sg.nperiod(), iper);

        // Compute the travel time for this source-receiver pair and period
        // using the surface wave dispersion grid (sg) and model grid (mg).
        // This is a placeholder; the actual computation would involve
        // ray tracing or finite-difference methods based on sg and mg.
        Eigen::MatrixX<real_t> svel_ij_inv = svel_ij.cwiseInverse();
        Eigen::MatrixX<real_t> svel_ij_inv_scaled = svel_ij_inv * _1_CR;
        Eigen::MatrixX<real_t> tfield = eikonal::FSM_UW_PS_lonlat_2d(
            mg.xgrids, mg.ygrids,
            m11_ij, m22_ij, -m12_ij,
            svel_ij_inv_scaled, evlo, evla
        );

        int n_rec = evt.second.rec_indices.size();
        Eigen::VectorXi idxs = Eigen::Map<const Eigen::VectorXi>(evt.second.rec_indices.data(), n_rec);
        Eigen::VectorX<real_t> stlo_rec = idxs.unaryExpr([&](int i){ return sr.stlo[i]; });
        Eigen::VectorX<real_t> stla_rec = idxs.unaryExpr([&](int i){ return sr.stla[i]; });
        Eigen::VectorX<real_t> tt_rec   = idxs.unaryExpr([&](int i){ return sr.tt[i]; });
        Eigen::VectorX<real_t> weight_rec = idxs.unaryExpr([&](int i){ return sr.weight[i]; });
        Eigen::VectorX<real_t> ttsyn = interp2d(mg.xgrids, mg.ygrids, tfield, stlo_rec, stla_rec);
        evt.second.syn_data = ttsyn;
        if (is_calc_adj) {
            // Compute the adjoint source time function for this event based on the travel time residuals and weights.
            Eigen::VectorX<real_t> adjoint_source = weight_rec.array() * (ttsyn - tt_rec).array();
            
            // Compute the adjoint field on the surface grid for this event using the adjoint source.
            Eigen::MatrixX<real_t> adj_field = eikonal::FSM_O1_JSE_lonlat_2d(
                mg.xgrids, mg.ygrids,
                m11_ij, m22_ij, -m12_ij,
                tfield, stlo_rec, stla_rec, adjoint_source
            );

            // Mask the adjoint field around the source location to avoid singularity and unrealistic large values.
            eikonal::mask_uniform_grid(mg.xgrids, mg.ygrids, adj_field, evlo, evla);
            
            // If density kernel is needed, compute and accumulate it as well.
            if (IP.postproc().is_kden) {
                Eigen::VectorX<real_t> adjoint_source_kden =
                    -weight_rec.array() * Eigen::VectorX<real_t>::Ones(n_rec).array();
                Eigen::MatrixX<real_t> kden_field = eikonal::FSM_O1_JSE_lonlat_2d(
                    mg.xgrids, mg.ygrids,
                    m11_ij, m22_ij, -m12_ij,
                    tfield, stlo_rec, stla_rec, adjoint_source_kden
                );
                eikonal::mask_uniform_grid(mg.xgrids, mg.ygrids, kden_field, evlo, evla);
                accumulate_kernels(sg, iper, adj_field, &kden_field, &tfield);
            } else {
                // Accumulate kernels for this event into the local accumulators in sg.
                accumulate_kernels(sg, iper, adj_field, nullptr, &tfield);
            }
            chi += 0.5 * (weight_rec.array() * (ttsyn - tt_rec).array().square()).sum();
        }
    }
    mpi.barrier();
    mpi.sum_all_all_inplace(chi);
    return chi;
}

void preproc::reset_kernel_accumulators( SurfGrid& sg) {
    auto& IP = InputParams::IP();

    if (run_mode == INVERSION_MODE || IP.inversion().model_para_type == MODEL_AZI_ANI) {
        // Reset the model perturbation arrays to zero before accumulating kernels.
        if (run_mode == INVERSION_MODE) {
            for (int iper = 0; iper < sg.nperiod(); ++iper) {
                sg.adj_s_local[iper].setZero();
                if (IP.postproc().is_kden) {
                    sg.kden_s_local[iper].setZero();
                }
                if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
                    sg.adj_xi_local[iper].setZero();
                    sg.adj_eta_local[iper].setZero();
                }
            }
        }
        sg.sen_vp_loc.setZero();
        sg.sen_vs_loc.setZero();
        sg.sen_rho_loc.setZero();
        if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
            sg.sen_gc_loc.setZero();
            sg.sen_gs_loc.setZero();
        }
    }
}

void preproc::prepare_dispersion_kernel(SurfGrid& sg) {
    auto& IP = InputParams::IP();
    auto& mpi = Parallel::mpi();

    if (run_mode == INVERSION_MODE || IP.inversion().model_para_type == MODEL_AZI_ANI) {
        sg.compute_dispersion_kernel();
        
        if (IP.topo().is_consider_topo) {
            sg.correct_depth_with_topo();
        }

        if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
            // TODO: need to be checked.
            sg.prepare_aniso_media();
        }
    }
    mpi.barrier();
}

void preproc::combine_kernels(SurfGrid& sg) {
    auto& IP = InputParams::IP();
    auto& mpi = Parallel::mpi();
    auto& logger = ATTLogger::logger();
    auto& dcp = Decomposer::DCP();
    auto& mg = ModelGrid::MG();


    // Resize to hold all parameter slots; default-construct (no storage) first,
    // then allocate only the entries that are actually used.
    logger.Info(fmt::format(
        "Combining traveltime kernels with surface wave ({}) kernels...", 
        waveTypeStr[static_cast<int>(sg.wave_type())]), MODULE_PREPROC);
    sg.ker_loc.assign(NPARAMS, Tensor3r());

    // vs kernel — always allocated (index 0)
    sg.ker_loc[0] = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
    sg.ker_loc[0].setZero();
    if (IP.inversion().use_alpha_beta_rho && sg.wave_type() == WaveType::RL) {
        // vp (1) and rho (2) — only when parametrised independently
        sg.ker_loc[1] = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        sg.ker_loc[1].setZero();
        sg.ker_loc[2] = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        sg.ker_loc[2].setZero();
    }
    if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
        // Gc (3) and Gs (4)
        sg.ker_loc[3] = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        sg.ker_loc[3].setZero();
        sg.ker_loc[4] = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        sg.ker_loc[4].setZero();
    }
    if (IP.postproc().is_kden){
        sg.ker_den_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        sg.ker_den_loc.setZero();
    }

    // Combine the local kernel accumulators across ranks to get the global kernel for each period, then apply the sensitivity kernels to get the model parameter kernels.
    for (int iper = 0; iper < sg.nperiod(); ++iper) {
        // Combine the local kernel accumulators across ranks to get the global kernel for this period.

        // reduce adj_s_local to the main rank
        Eigen::MatrixX<real_t> adj_tt = Eigen::MatrixX<real_t>::Zero(ngrid_i, ngrid_j);
        Eigen::MatrixX<real_t> adj_den, adj_xi, adj_eta;
        mpi.sum_all_all(sg.adj_s_local[iper].data(), adj_tt.data(), ngrid_i * ngrid_j);
        if ( IP.postproc().is_kden ) {
            adj_den = Eigen::MatrixX<real_t>::Zero(ngrid_i, ngrid_j);
            mpi.sum_all_all(sg.kden_s_local[iper].data(), adj_den.data(), ngrid_i * ngrid_j);
        }
        if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
            adj_xi  = Eigen::MatrixX<real_t>::Zero(ngrid_i, ngrid_j);
            adj_eta = Eigen::MatrixX<real_t>::Zero(ngrid_i, ngrid_j);
            mpi.sum_all_all(sg.adj_xi_local[iper].data(),  adj_xi.data(),  ngrid_i * ngrid_j);
            mpi.sum_all_all(sg.adj_eta_local[iper].data(), adj_eta.data(), ngrid_i * ngrid_j);
        }

        // Isotropic parameter kernels — gated by use_alpha_beta_rho only
        if (IP.inversion().use_alpha_beta_rho && sg.wave_type() == WaveType::RL) {
            for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
                for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
                    const int iglob_x = dcp.loc_I_start() + ix;
                    const int iglob_y = dcp.loc_J_start() + iy;
                    const real_t att = adj_tt(iglob_x, iglob_y);
                    // In the anisotropic case term1 = -1/sqrt(1+r1²+r2²), otherwise 1
                    const real_t r1 = IP.inversion().model_para_type == MODEL_AZI_ANI ? sg.r1_loc(ix, iy, iper) : _0_CR;
                    const real_t r2 = IP.inversion().model_para_type == MODEL_AZI_ANI ? sg.r2_loc(ix, iy, iper) : _0_CR;
                    const real_t scale = _1_CR / std::sqrt(_1_CR + r1*r1 + r2*r2);
                    for (int k = 0; k < ngrid_k; ++k) {
                        sg.ker_loc[0](ix, iy, k) -= att * scale * sg.sen_vs_loc(ix, iy, k, iper);
                        sg.ker_loc[1](ix, iy, k) -= att * scale * sg.sen_vp_loc(ix, iy, k, iper);
                        if (IP.postproc().is_kden) {
                            sg.ker_den_loc(ix, iy, k) -= adj_den(iglob_x, iglob_y) * scale * sg.sen_vs_loc(ix, iy, k, iper);
                        }
                        if (!IP.inversion().rho_scaling) {
                            sg.ker_loc[2](ix, iy, k) -= att * scale * sg.sen_rho_loc(ix, iy, k, iper);
                        }
                    }
                }
            }
            if (IP.inversion().rho_scaling) {
                sg.ker_loc[2] = sg.ker_loc[0] * RHO_SCALING;
            }
        } else if (sg.wave_type() == WaveType::RL) {
            // vs-only parametrisation: chain rule collapses vp and rho into the vs kernel
            //   K_vs += adj_s * (sen_vs + sen_vp * d(vp)/d(vs) + sen_rho * d(rho)/d(vp) * d(vp)/d(vs))
            for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
                for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
                    const int iglob_x = dcp.loc_I_start() + ix;
                    const int iglob_y = dcp.loc_J_start() + iy;
                    const real_t att  = adj_tt(iglob_x, iglob_y);
                    for (int k = 0; k < ngrid_k; ++k) {
                        const real_t vs  = mg.vs3d_loc(ix, iy, k);
                        const real_t vp  = mg.vp3d_loc(ix, iy, k);
                        const real_t dab = dalpha_dbeta(vs);   // d(vp)/d(vs)
                        const real_t dra = drho_dalpha(vp);    // d(rho)/d(vp)
                        sg.ker_loc[0](ix, iy, k) -= att * (
                            sg.sen_vs_loc(ix, iy, k, iper)
                            + sg.sen_vp_loc(ix, iy, k, iper)  * dab
                            + sg.sen_rho_loc(ix, iy, k, iper) * dra * dab
                        );
                        if (IP.postproc().is_kden) {
                            sg.ker_den_loc(ix, iy, k) -= adj_den(iglob_x, iglob_y) * (
                                sg.sen_vs_loc(ix, iy, k, iper)
                                + sg.sen_vp_loc(ix, iy, k, iper)  * dab
                                + sg.sen_rho_loc(ix, iy, k, iper) * dra * dab
                            );
                        }
                    }
                }
            }
        } else {
            // For Love waves with no independent vp kernel, the vs kernel includes contributions from the vp and rho kernels through the chain rule, but without the scaling factor that appears in the anisotropic case.
            for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
                for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
                    const int iglob_x = dcp.loc_I_start() + ix;
                    const int iglob_y = dcp.loc_J_start() + iy;
                    const real_t att  = adj_tt(iglob_x, iglob_y);
                    for (int k = 0; k < ngrid_k; ++k) {
                        sg.ker_loc[0](ix, iy, k) -= att * sg.sen_vs_loc(ix, iy, k, iper);
                        if (IP.postproc().is_kden) {
                            sg.ker_den_loc(ix, iy, k) -= adj_den(iglob_x, iglob_y) * sg.sen_vs_loc(ix, iy, k, iper);
                        }
                    }
                }
            }
        }

        // Anisotropic parameter kernels (Gc, Gs)
        if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
            for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
                for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
                    const int iglob_x = dcp.loc_I_start() + ix;
                    const int iglob_y = dcp.loc_J_start() + iy;
                    const real_t r1    = sg.r1_loc(ix, iy, iper);
                    const real_t r2    = sg.r2_loc(ix, iy, iper);
                    const real_t D     = _1_CR + r1*r1 + r2*r2;
                    const real_t sqrtD = std::sqrt(D);
                    const real_t D15   = D * sqrtD;   // D^1.5
                    const real_t D2    = D * D;
                    const real_t sv    = sg.svel[sg.surf_idx(iglob_x, iglob_y, iper)];
                    const real_t att   = adj_tt(iglob_x, iglob_y);
                    const real_t axi   = adj_xi(iglob_x, iglob_y);
                    const real_t aeta  = adj_eta(iglob_x, iglob_y);
                    for (int k = 0; k < ngrid_k; ++k) {
                        // Gc kernel: term2·r1 + term3·adj_xi + term4·adj_eta
                        sg.ker_loc[3](ix, iy, k) += (
                            - att * r1 / D15
                            + axi  * (_1_CR - r1*r1 + r2*r2) / (sv * D2)
                            + aeta * (-_2_CR * r1 * r2)       / (sv * D2)
                        ) * sg.sen_gc_loc(ix, iy, k, iper);
                        // Gs kernel: term2·r2 + term4·adj_xi + term5·adj_eta
                        sg.ker_loc[4](ix, iy, k) += (
                            - att * r2 / D15
                            + axi  * (-_2_CR * r1 * r2)      / (sv * D2)
                            + aeta * (_1_CR + r1*r1 - r2*r2) / (sv * D2)
                        ) * sg.sen_gs_loc(ix, iy, k, iper);
                    }
                }
            }
        }
    }
    mpi.barrier();
}
    

