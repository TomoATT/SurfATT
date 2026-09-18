#include "inversion.h"
#include "logger.h"
#include "config.h"
#include "utils.h"
#include "h5io.h"
#include "src_rec.h"
#include "xdmf.h"
#include <cmath>
#include <fstream>

static void distribute_model_para(){
    auto& mg = ModelGrid::MG();
    auto& dcp = Decomposer::DCP();
    auto& IP = InputParams::IP();

    // Distribute the global model to per-rank local slices before fwdsurf().
    mg.vs3d_loc = dcp.distribute_data(mg.vs3d);
    mg.vp3d_loc = dcp.distribute_data(mg.vp3d);
    mg.rho3d_loc = dcp.distribute_data(mg.rho3d);
    if (IP.inversion().model_para_type == MODEL_AZI_ANI) {  // azimuthal anisotropy
        mg.gc3d_loc = dcp.distribute_data(mg.gc3d);
        mg.gs3d_loc = dcp.distribute_data(mg.gs3d);
    }
    if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {  // radial anisotropy
        mg.vsh3d_loc = dcp.distribute_data(mg.vsh3d);
        mg.gamma3d_loc = mg.vsh3d_loc / mg.vs3d_loc;  // convert to gamma for radial anisotropy parameterization
    }
}

Inversion::Inversion() {
    auto &dcp = Decomposer::DCP();
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();
    auto &mpi = Parallel::mpi();

    // initialize gradient
    if (run_mode == INVERSION_MODE) {
        // initialize HDF5 file for storing model and gradient history (used by LBFGS)
        db_fname = fmt::format("{}/{}", IP.output().output_path, MODEL_ITER_FNAME);

        if (mpi.is_main()) {
            try {
                auto &mg = ModelGrid::MG();
                H5IO f(db_fname, H5IO::TRUNC);
                f.write_vector("x", mg.xgrids);
                f.write_vector("y", mg.ygrids);
                f.write_vector("z", mg.zgrids);

                // Uniform-scale coordinates: use an equirectangular projection
                // about the grid's central latitude.  A 3DRectMesh requires each
                // coordinate axis to be a separate 1-D vector, so the longitude
                // scale factor must be a scalar rather than one value per latitude.
                const real_t ref_lat = _0_5_CR *
                    (mg.ygrids(0) + mg.ygrids(mg.ygrids.size() - 1));
                const real_t cos_ref_lat = std::cos(ref_lat * DEG2RAD);
                Eigen::VectorX<real_t> x_km =
                    (mg.xgrids.array() - mg.xgrids(0)) * DEG2RAD * R_EARTH * cos_ref_lat;
                Eigen::VectorX<real_t> y_km =
                    (mg.ygrids.array() - mg.ygrids(0)) * DEG2RAD * R_EARTH;
                f.write_vector("x_km", x_km);
                f.write_vector("y_km", y_km);
            } catch (const std::exception &e) {
                logger.Error(fmt::format("Failed to create HDF5 file for model history: {}", e.what()), MODULE_INV);
                logger.Error("Check if the output path exists and is writable, or delete the existing file.", MODULE_INV);
                mpi.abort(EXIT_FAILURE);
            }
        }
        xdmf_fname_ = fmt::format("{}/model_iter.xdmf", IP.output().output_path);

        // Create empty datasets for model and gradient history. These will be resized and filled during the inversion iterations.
        is_active_param[0] = true; // vs is always active
        is_active_param[1] = IP.inversion().use_alpha_beta_rho; // vp active if use_alpha_beta_rho is true
        is_active_param[2] = IP.inversion().use_alpha_beta_rho; // rho active if use_alpha_beta_rho is true
        is_active_param[3] = IP.inversion().model_para_type == MODEL_AZI_ANI; // gc active if model_para_type is azimuthal anisotropy
        is_active_param[4] = IP.inversion().model_para_type == MODEL_AZI_ANI; // gs active if model_para_type is azimuthal anisotropy
        is_active_param[5] = IP.inversion().model_para_type == MODEL_RADIAL_ANI; // gamma active if model_para_type is radial anisotropy
        gradient_.assign(NPARAMS, Tensor3r());
        for (int p = 0; p < NPARAMS; ++p) {
            if (is_active_param[p]) {
                gradient_[p] = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
            }
        }
        if (IP.inversion().optim_method == OPTIM_LBFGS) {
            ker_curr_.assign(NPARAMS, Tensor3r());
            ker_prev_.assign(NPARAMS, Tensor3r());
            for (int p = 0; p < NPARAMS; ++p) {
                if (is_active_param[p]) {
                    ker_curr_[p] = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
                    ker_prev_[p] = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
                    ker_curr_[p].setZero();
                    ker_prev_[p].setZero();
                }
            }
        } else if (IP.inversion().optim_method == OPTIM_SD) {
            gradient_prev_.assign(NPARAMS, Tensor3r());
            for (int p = 0; p < NPARAMS; ++p) {
                if (is_active_param[p]) {
                    gradient_prev_[p] = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
                    gradient_prev_[p].setZero();
                }
            }
        } else {
            logger.Error("Unsupported optimization method specified in input parameters.", MODULE_INV);
            mpi.abort(EXIT_FAILURE);
        }

        // Set the initial step length for the optimization. This can be tuned or made adaptive in the future.
        alpha_ = IP.inversion().step_length;

        // Open objective function log on main rank
        if (mpi.is_main() && run_mode == INVERSION_MODE) {
            const std::string obj_path = fmt::format("{}/{}", IP.output().output_path, OBJ_FNAME);
            obj_file_.open(obj_path);
            if (!obj_file_) {
                logger.Error(fmt::format("Cannot open {} for writing", obj_path), MODULE_INV);
                mpi.abort(EXIT_FAILURE);
            } else {
                obj_file_ << std::unitbuf;  // flush after every write operation
                obj_file_ << fmt::format(
                    "{:<6} {:>14} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}\n",
                    "iter", "misfit",
                    "res_rl_ph_mean", "res_rl_ph_std", "res_rl_gr_mean", "res_rl_gr_std",
                    "res_lv_ph_mean", "res_lv_ph_std", "res_lv_gr_mean", "res_lv_gr_std",
                    "step_length"
                );
            }
        }
    }
};

void Inversion::run_forward() {
    auto& logger = ATTLogger::logger();
    logger.Info("Running forward calculation...", MODULE_INV);
    distribute_model_para();
    run_forward_adjoint(false);
    write_src_rec_fwd();
}

void Inversion::write_obj_line()
{
    auto &mpi = Parallel::mpi();
    auto &IP  = InputParams::IP();

    // compute_residual_stats() uses MPI allreduce — all ranks must call it
    // the same number of times, so we iterate every active (wt, vt) entry.
    ResidualStats rl_ph_stats{_0_CR, _0_CR};
    ResidualStats rl_gr_stats{_0_CR, _0_CR};
    ResidualStats lv_ph_stats{_0_CR, _0_CR};
    ResidualStats lv_gr_stats{_0_CR, _0_CR};
    for (auto [wt, vt] : IP.data().active_data) {
        auto stats = SrcRec::SR(wt, vt).compute_residual_stats();
        if (wt == WaveType::RL && vt == SurfType::PH) {
            rl_ph_stats = stats;
        } else if (wt == WaveType::RL && vt == SurfType::GR) {
            rl_gr_stats = stats;
        } else if (wt == WaveType::LV && vt == SurfType::PH) {
            lv_ph_stats = stats;
        } else if (wt == WaveType::LV && vt == SurfType::GR) {
            lv_gr_stats = stats;
        }
    }

    if (mpi.is_main() && obj_file_)
        obj_file_ << fmt::format(
            "{:<6d} {:>14.6e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.6f}\n",
            iter_, misfit_[iter_],
            rl_ph_stats.mean, rl_ph_stats.stddev, rl_gr_stats.mean, rl_gr_stats.stddev,
            lv_ph_stats.mean, lv_ph_stats.stddev, lv_gr_stats.mean, lv_gr_stats.stddev,
            alpha_
        );
}

void Inversion::run_inversion() {
    auto &IP = InputParams::IP();
    auto &mpi = Parallel::mpi();
    auto &mg = ModelGrid::MG();
    auto &logger = ATTLogger::logger();

    for ( iter_ = 0; iter_ < IP.inversion().niter; ++iter_ ) {
        logger.Info(fmt::format("Starting inversion {}th iteration ...", iter_), MODULE_INV);
        // Initialize the iteration: distribute the current model to local subdomains, reset model update and search direction
        init_iteration();

        // Run forward and adjoint calculations to compute the gradient.
        // When using L-BFGS, the accepted line-search step already computed g_{k+1}
        // and stored it in gradient_ / ker_curr_, so we can skip this call.
        const bool need_fwd_adj = !gradient_reuse_;  // flag reset inside init_iteration()
        if (need_fwd_adj) {
            misfit_trial_ = run_forward_adjoint(true);
            write_src_rec_fwd();
        } else {
            logger.Info("Reusing gradient from accepted line-search step (skipping forward+adjoint).", MODULE_INV);
        }
        misfit_[iter_] = misfit_trial_;
        logger.Info(fmt::format(
            "Completed inversion {}th iteration with misfit = {:.4f} ({:.2f}%)", iter_, misfit_[iter_],
                100 * misfit_[iter_] / misfit_[0]
        ), MODULE_INV);

        write_obj_line();

        if (IP.postproc().is_kden) {
            store_kernel_density();
        }

        if (IP.output().output_in_process_model || IP.inversion().optim_method == OPTIM_LBFGS) {
            store_gradient();
        }

        // Update the model using the computed gradient
        if ( IP.inversion().optim_method == OPTIM_SD ) {
            steepest_descent();
        } else if (IP.inversion().optim_method == OPTIM_LBFGS) {
            logger.Info("Using L-BFGS optimization with line search.", MODULE_INV);
            while (true) {
                if ( !line_search() ) break;
                if (wolfe_res_.status == optimize::WolfeResult::Status::FAIL) break;
                iter_start_ = iter_;
                logger.Info(fmt::format(
                    "Restarting count of L-BFGS from {:03d}", iter_start_
                ), MODULE_INV);
            }
            mg.collect_model_loc();
            write_src_rec_fwd();
        } else {
            logger.Error("Unsupported optimization method specified in input parameters.", MODULE_INV);
            mpi.abort(EXIT_FAILURE);
        }

        // Wolfe status is only valid for the L-BFGS path after line_search().
        const bool line_search_failed =
            IP.inversion().optim_method == OPTIM_LBFGS &&
            wolfe_res_.status == optimize::WolfeResult::Status::FAIL;

        // Check for convergence based on misfit reduction
        if (check_convergence() || line_search_failed) break;

        mpi.barrier();
    }
    mg.write(FINAL_MODEL_FNAME);
    write_src_rec_fwd();
}

void Inversion::init_iteration() {
    auto& IP = InputParams::IP();
    auto& mpi = Parallel::mpi();

    if (IP.output().output_in_process_model || IP.inversion().optim_method == OPTIM_LBFGS) {
        store_model();
        mpi.barrier();
    }

    // Distribute the updated global model to per-rank local slices before fwdsurf().
    distribute_model_para();

    // initialize kernel for current and previous iteration to zero
    if (IP.inversion().optim_method == OPTIM_LBFGS){
        if (!gradient_reuse_) {
            // Fresh start: zero everything
            for (int p = 0; p < NPARAMS; ++p) {
                if (is_active_param[p]) {
                    ker_curr_[p].setZero();
                    ker_prev_[p].setZero();
                }
            }
        } else {
            // gradient_ and ker_curr_ are already valid (carried from accepted line search).
            // Only zero ker_prev_; line_search() will repopulate it via ker_prev_ = ker_curr_.
            for (int p = 0; p < NPARAMS; ++p)
                if (is_active_param[p]) ker_prev_[p].setZero();
        }
    }

    // Initialize gradient to zero — skip when reusing from accepted line search
    if (!gradient_reuse_) {
        for (int ipara = 0; ipara < NPARAMS; ++ipara)
            if (is_active_param[ipara]) gradient_[ipara].setZero();
    }
    gradient_reuse_ = false;  // consumed; will be set again by line_search on next accept
    mpi.barrier();
}

bool Inversion::check_convergence() {
    auto &mpi = Parallel::mpi();
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();

    if (mpi.is_main()) {
        if ( iter_ > BREAK_ITER ) {
            real_t sum_misfit_prev = _0_CR;
            real_t sum_misfit_curr = _0_CR;

            for (int i = 0; i <= BREAK_ITER; ++i) {
                sum_misfit_prev += misfit_[iter_ - 1 - i];
                sum_misfit_curr += misfit_[iter_ - i];
            }
            sum_misfit_prev /= BREAK_ITER;
            sum_misfit_curr /= BREAK_ITER;
            real_t misfit_reduction = 100 * (sum_misfit_prev - sum_misfit_curr) / sum_misfit_prev;
            if (misfit_reduction < 0) {
                logger.Info(fmt::format("Misfit increased in the last {} iterations.", BREAK_ITER), MODULE_INV);
                break_flag_ = true;
            } else if (misfit_reduction < IP.inversion().min_derr) {
                logger.Info(fmt::format("Convergence achieved with misfit reduction of {:.2f}% in the last {} iterations.", misfit_reduction, BREAK_ITER), MODULE_INV);
                break_flag_ = true;
            } else {
                logger.Info(fmt::format("Misfit reduction of {:.2f}% in the last {} iterations.", misfit_reduction, BREAK_ITER), MODULE_INV);
                break_flag_ = false;
            }
        }
    }
    mpi.bcast(break_flag_);
    return break_flag_;
}

void Inversion::accumulate_smoothed_gradient(
    WaveType wt,
    int itype,
    real_t chi,
    const FieldVec &ker_smooth
) {
    auto &IP = InputParams::IP();
    const int model_para_type = IP.inversion().model_para_type;
    const real_t grad_scale = IP.data().weights[itype] / chi;
    auto accumulate_grad = [&](int ipara, bool check_active = true) {
        if (!check_active || is_active_param[ipara]) {
            gradient_[ipara] += ker_smooth[ipara] * grad_scale;
        }
    };

    if (model_para_type == MODEL_RADIAL_ANI) {
        if (wt == WaveType::RL) {
            for (int ipara = 0; ipara < 3; ++ipara) {
                accumulate_grad(ipara, true);
            }
        } else if (wt == WaveType::LV) {
            // Keep previous behavior: gamma term is always accumulated for Love in radial anisotropy.
            accumulate_grad(NPARAMS - 1, false);
        }
        return;
    }

    const int ipara_end = (model_para_type == MODEL_AZI_ANI) ? (NPARAMS - 1) : NPARAMS;
    for (int ipara = 0; ipara < ipara_end; ++ipara) {
        accumulate_grad(ipara, true);
    }
}

real_t Inversion::run_forward_adjoint(const bool is_calc_adj) {
    auto& IP = InputParams::IP();
    auto& mpi = Parallel::mpi();

    real_t misfit_total = _0_CR;
    for (auto [wt, tp] : IP.data().active_data) {
        int itype = static_cast<int>(tp);
        auto &sg = SurfGrid::SG(wt, tp);
        auto &sr = SrcRec::SR(wt, tp);

        // Reset kernel accumulators before processing this type of data
        preproc::reset_kernel_accumulators(sg);

        // Compute surface wave dispersion from the 3D S-wave velocity model.
        sg.fwdsurf();

        // Compute the dispersion kernel (sensitivity of travel times to
        //   velocity perturbations at each surface grid point) for this type of data.
        // Kernels are only needed for the adjoint (gradient) calculation.
        preproc::prepare_dispersion_kernel(sg);

        // calculate travel time for each source-receiver pair and period, and store in sr.events_local
        real_t chi = preproc::forward_for_event(sr, sg, is_calc_adj);
        
        if (run_mode == INVERSION_MODE) {
            misfit_total += chi * IP.data().weights[itype];

            // Combine the local kernel accumulators across ranks to get the global kernel for each period, then apply the sensitivity kernels to get the model parameter kernels.
            preproc::combine_kernels(sg);

            //backup the current kernel before preconditioning and smoothing (used for LBFGS)
            if (IP.inversion().optim_method == OPTIM_LBFGS) {
                const real_t wt_val = IP.data().weights[itype];
                if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
                    // For radial anisotropy, Love kernel lives in ker_loc[0] and contributes
                    // to the gamma (index 5) direction; Rayleigh only updates vs/vp/rho.
                    if (wt == WaveType::RL) {
                        for (int ipara = 0; ipara < 3; ++ipara)
                            if (is_active_param[ipara])
                                ker_curr_[ipara] = ker_curr_[ipara] + sg.ker_loc[ipara] * wt_val;
                    } else {
                        ker_curr_[5] = ker_curr_[5] + sg.ker_loc[0] * wt_val;
                    }
                } else {
                    for (int ipara = 0; ipara < NPARAMS; ++ipara)
                        if (is_active_param[ipara])
                            ker_curr_[ipara] = ker_curr_[ipara] + sg.ker_loc[ipara] * wt_val;
                }
            }

            // apply preconditioning to the kernels if needed
            postproc::kernel_precondition(sg);

            // smooth the kernels if needed
            auto ker_smooth = postproc::kernel_smooth(sg);

            accumulate_smoothed_gradient(wt, itype, chi, ker_smooth);
        }
        mpi.barrier();
    }
    if (run_mode == INVERSION_MODE && IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
        convert_radial_kl();
        mpi.barrier();
    }
    return misfit_total;
}

void Inversion::grad_normalization(FieldVec &grads) {
    auto& mpi = Parallel::mpi();

    real_t local_max = _0_CR;
    for (int ipara = 0; ipara < NPARAMS; ++ipara) {
        if (is_active_param[ipara]) {
            Eigen::Tensor<real_t, 0, Eigen::RowMajor> max_tensor = grads[ipara].abs().maximum();
            local_max = std::max(local_max, max_tensor());
        }
    }

    real_t global_max;
    mpi.max_all_all(local_max, global_max);
    if (global_max < REAL_EPS) return;  // guard against zero gradient
    for (int ipara = 0; ipara < NPARAMS; ++ipara) {
        if (is_active_param[ipara]) {
            grads[ipara] = grads[ipara] / global_max;
        }
    }
}

// Save the current model parameters (before update) to db_fname.
// Dataset names: model_vs_{N}, model_vp_{N}, etc.
// mg.vs3d etc. are the global arrays already on all ranks; only main rank writes.
void Inversion::store_model() {
    auto &mg  = ModelGrid::MG();
    auto &IP  = InputParams::IP();
    auto &mpi = Parallel::mpi();

    if (!mpi.is_main()) return;

    H5IO f(db_fname, H5IO::RDWR);
    const std::string sfx = fmt::format("_{:03d}", iter_);

    // Use TensorMap to wrap the raw global pointer without copying
    using TMap = Eigen::TensorMap<Tensor3r>;
    f.write_tensor("model_vs" + sfx, TMap(mg.vs3d,  ngrid_i, ngrid_j, ngrid_k));
    if (IP.inversion().use_alpha_beta_rho) {
        f.write_tensor("model_vp"  + sfx, TMap(mg.vp3d,  ngrid_i, ngrid_j, ngrid_k));
        f.write_tensor("model_rho" + sfx, TMap(mg.rho3d, ngrid_i, ngrid_j, ngrid_k));
    }
    if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
        f.write_tensor("model_gc" + sfx, TMap(mg.gc3d, ngrid_i, ngrid_j, ngrid_k));
        f.write_tensor("model_gs" + sfx, TMap(mg.gs3d, ngrid_i, ngrid_j, ngrid_k));
    }
    if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
        // vsh3d and vs3d are already in global memory (collected by collect_model_loc)
        TMap vsh_map(mg.vsh3d, ngrid_i, ngrid_j, ngrid_k);
        TMap vs_map(mg.vs3d, ngrid_i, ngrid_j, ngrid_k);
        Tensor3r gamma3d = vsh_map / vs_map;
        f.write_tensor("model_gamma" + sfx, gamma3d);
    }
    // Gradients for iterations 0..iter_-1 have been written; current iter_ grad
    // is written later by store_gradient(), so last_grad_iter = iter_ - 1.
    const bool grads_enabled = IP.output().output_in_process_model
                                || IP.inversion().optim_method == OPTIM_LBFGS;
    xdmf::write_model_iter(xdmf_fname_, iter_, grads_enabled ? iter_ - 1 : -1);
}

// Save the current (normalised) gradient to db_fname.
// Dataset names: grad_vs_iter{N}, grad_vp_iter{N}, etc.
void Inversion::store_gradient() {
    auto &dcp = Decomposer::DCP();
    auto &mpi = Parallel::mpi();

    // All ranks participate in the gathering of the gradient
    FieldVec grad_all(NPARAMS);
    for (int i = 0; i < NPARAMS; ++i) {
        if (is_active_param[i]) 
            grad_all[i] = dcp.collect_data(gradient_[i].data());
    }

    if (!mpi.is_main()) return;

    H5IO f(db_fname, H5IO::RDWR);
    const std::string sfx = fmt::format("_{:03d}", iter_);

    for (int i = 0; i < NPARAMS; ++i) {
        if (is_active_param[i])
            f.write_tensor(std::string("grad_") + pnames[i] + sfx, grad_all[i]);
    }
    // grad_vs_{iter_} is now in HDF5; update XDMF so all stored iterations
    // (0..iter_) show both model and gradient fields.
    if (!xdmf_fname_.empty())
        xdmf::write_model_iter(xdmf_fname_, iter_, iter_);
}

// Save the total kernel density used during the current inversion iteration.
// Each active data type contributes according to its phase/group data weight.
// Dataset names follow the model/gradient convention: kernel_density_{N}.
void Inversion::store_kernel_density() {
    auto &IP  = InputParams::IP();
    auto &dcp = Decomposer::DCP();
    auto &mpi = Parallel::mpi();

    Tensor3r density_loc(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
    density_loc.setZero();
    for (auto [wt, tp] : IP.data().active_data) {
        const int itype = static_cast<int>(tp);
        density_loc += SurfGrid::SG(wt, tp).ker_den_loc * IP.data().weights[itype];
    }

    // collect_data is collective: all ranks participate, then only the main
    // rank owns and writes the assembled global volume.
    Tensor3r density_all = dcp.collect_data(density_loc.data());
    if (!mpi.is_main()) return;

    H5IO f(db_fname, H5IO::RDWR);
    f.write_tensor(fmt::format("kernel_density_{:03d}", iter_), density_all);
}

void Inversion::steepest_descent() {
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();
    auto &mg = ModelGrid::MG();

    grad_normalization(gradient_);

    logger.Info(fmt::format("Steepest descent optimization with step length {:.6f}", alpha_), MODULE_INV);
    if (iter_ > 0) {
        bool shrink_flag = false;
        real_t decs_angle = optimize::calc_descent_angle(gradient_prev_, gradient_);
        if (decs_angle > MAX_SD_ANGLE ) {
            shrink_flag = true;
            logger.Info(fmt::format("Descent angle {:.4f} exceeds maximum {:.2f}", decs_angle, MAX_SD_ANGLE), MODULE_INV);
        } else if (misfit_[iter_] > misfit_[iter_ - 1]) {
            shrink_flag = true;
            logger.Info(fmt::format("Misfit increased from {:.4f} to {:.4f}", misfit_[iter_ - 1], misfit_[iter_]), MODULE_INV);
        }
        if (shrink_flag) {
            alpha_ *= IP.inversion().maxshrink;
            logger.Info(fmt::format("Shrinking step length to {:.6f}", alpha_), MODULE_INV);
        }
    }
    // copy the current gradient to the previous gradient for potential future use
    gradient_prev_ = gradient_;

    // Pass gradient directly; model_update applies model -= alpha * gradient (descent).
    model_update(gradient_);
    mg.collect_model_loc();  // gather the updated local model back to the global model
}

bool Inversion::line_search() {
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();
    auto &mpi = Parallel::mpi();
    bool break_sub_iter = false, restart_flag = false;
    real_t misfit_trial = _0_CR;

    logger.Info("Optimization with L-BFGS method", MODULE_INV);

    FieldVec search_dir(NPARAMS);
    if (iter_ == iter_start_) {
        // First iteration: use gradient as direction (model_update subtracts it)
        search_dir = gradient_;
    } else {
        // Compute L-BFGS search direction based on current and previous gradients and model updates
        search_dir = optimize::lbfgs_direction(iter_);
    }

    grad_normalization(search_dir);

    if ( iter_ > iter_start_ ) {
        real_t desc_angle = optimize::calc_descent_angle(search_dir, gradient_);
        logger.Info(fmt::format("Angle between direction and negative gradient: {:.4f} degrees",
            desc_angle), MODULE_INV
        );
        if (desc_angle > MAX_DESC_ANGLE) {
            restart_flag = true;
            return restart_flag;
        }
    } else {
        alpha_ = IP.inversion().step_length; // use initial step length for the first iteration
    }

    ker_prev_ = ker_curr_;

    int sub_iter = 0;
    alpha_R_ = _0_CR;
    alpha_L_ = _0_CR;
    for ( sub_iter = 0; sub_iter < IP.inversion().max_sub_niter; ++sub_iter ) {
        logger.Info(fmt::format(
            "Line search sub-iteration {}: testing step length alpha = {:.6f}", sub_iter, alpha_
        ), MODULE_INV);

        // Reset kernel and gradient accumulators for each independent trial.
        // ker_curr_ must reflect only the current alpha, not a sum over all trials.
        for (int p = 0; p < NPARAMS; ++p) {
            if (is_active_param[p]) {
                ker_curr_[p].setZero();
                gradient_[p].setZero();
            }
        }

        // Reset local model slices from the global (pre-line-search) model before
        // each trial step, so sub-iterations are independent of each other.
        distribute_model_para();
        model_update(search_dir);

        // Run forward calculation with the updated model to evaluate the misfit at this step length
        misfit_trial = run_forward_adjoint(true);

        // wolfe_condition checks if the current step length satisfies the strong Wolfe conditions and returns the next step length to try if not.
        wolfe_res_ = optimize::wolfe_condition(
            ker_prev_, ker_curr_,
            search_dir, alpha_, alpha_L_, alpha_R_,
            misfit_[iter_], misfit_trial, sub_iter
        );

        if (wolfe_res_.status == optimize::WolfeResult::Status::ACCEPT) {
            logger.Info(fmt::format("Line search accepted with alpha = {:.6f}", alpha_), MODULE_INV);
            break_sub_iter = true;
        } else if (wolfe_res_.status == optimize::WolfeResult::Status::TRY) {
            alpha_ = wolfe_res_.next_alpha;
            logger.Info(fmt::format("Line search trying next alpha = {:.6f}", alpha_), MODULE_INV);
            break_sub_iter = false;
        } else {
            logger.Info("Line search failed to find a suitable step length.", MODULE_INV);
            break_sub_iter = true;
            restart_flag = true;
        }
        mpi.barrier();
        if (break_sub_iter) break;
    }
    if (!restart_flag) misfit_trial_ = misfit_trial;
    return restart_flag;
}

void Inversion::model_update(FieldVec &dir) {
    auto &IP = InputParams::IP();
    auto &mg = ModelGrid::MG();
    auto &mpi = Parallel::mpi();

    // dir is the gradient direction; apply model -= alpha * dir (gradient descent).
    // For vs/vp/rho the update is multiplicative (log-space additive): model *= (1 - alpha * dir).
    mg.vs3d_loc = mg.vs3d_loc * (1 - alpha_ * dir[0]);
    if (IP.inversion().use_alpha_beta_rho) {
        mg.vp3d_loc = mg.vp3d_loc * (1 - alpha_ * dir[1]);
        mg.rho3d_loc = mg.rho3d_loc * (1 - alpha_ * dir[2]);
        alpha_clamp();
    } else {
        // Empirical scaling: vs → vp → rho via Brocher (2005)
        mg.vp3d_loc  = vs2vp(mg.vs3d_loc);
        mg.rho3d_loc = vp2rho(mg.vp3d_loc);
    }
    if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
        mg.gc3d_loc = mg.gc3d_loc - alpha_ * dir[3];
        mg.gs3d_loc = mg.gs3d_loc - alpha_ * dir[4];
    }
    if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
        mg.gamma3d_loc = mg.gamma3d_loc * (1 - alpha_ * dir[5]);
        mg.vsh3d_loc = mg.vs3d_loc * mg.gamma3d_loc;
    }
    mpi.barrier();
}

void Inversion::alpha_clamp() {
    auto &IP = InputParams::IP();
    auto &mg = ModelGrid::MG();

    const real_t ratio_min = IP.inversion().vpvs_ratio_range[0];
    const real_t ratio_max = IP.inversion().vpvs_ratio_range[1];

    // Clamp vp/vs ratio to [ratio_min, ratio_max]
    mg.vp3d_loc = mg.vp3d_loc
        .cwiseMax(ratio_min * mg.vs3d_loc)
        .cwiseMin(ratio_max * mg.vs3d_loc);
}

void Inversion::write_src_rec_fwd(){
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();

    for (auto [wt, tp] : IP.data().active_data) {
        int itype = static_cast<int>(tp);
        auto &sr = SrcRec::SR(wt, tp);
        const std::string tag = waveTypeStr[static_cast<int>(wt)] + "_" + surfTypeStr[itype];
        // gather synthetic travel times to the main rank for output and inversion steps
        if (run_mode == FORWARD_ONLY || IP.output().output_in_process_data ||
            (run_mode == INVERSION_MODE && iter_ == IP.inversion().niter - 1) ||
            (run_mode == INVERSION_MODE && iter_ == 0) ||
            wolfe_res_.status == optimize::WolfeResult::Status::FAIL||
            break_flag_) {
            logger.Info(fmt::format(
                "Gathering synthetic {} travel times to the main rank for output...", tag), MODULE_PREPROC
            );
            sr.gather_syn_tt();
            std::string sfx = tag;
            if (run_mode == INVERSION_MODE)
                sfx = fmt::format("{}_{:03d}", tag, iter_);
            sr.write(
                fmt::format("{}/{}_{}.csv", IP.output().output_path, FORWARD_FILE_PREFIX, sfx), true
            );
        }
    }
}
