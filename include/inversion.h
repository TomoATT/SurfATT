#pragma once

#include "config.h"
#include "parallel.h"
#include "input_params.h"
#include "logger.h"
#include "model_grid.h"
#include "surf_grid.h"
#include "decomposer.h"
#include "preproc.h"
#include "postproc.h"
#include "optimize.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <memory>
#include <stdexcept>


class Inversion {
public:
    static void init() {
        get_instance_ptr() = std::make_unique<Inversion>();
    }
    static Inversion& INV() {
        auto* ptr = get_instance_ptr().get();
        if (!ptr) throw std::runtime_error("Inversion: call init() first");
        return *ptr;
    }

    Inversion(const Inversion&)            = delete;
    Inversion& operator=(const Inversion&) = delete;

    Inversion();
    ~Inversion() = default;

    // Run only the forward calculation (no kernel accumulation or model update).
    void run_forward();

    // Run one full inversion iteration: forward + adjoint + kernel combination + model update.
    void run_inversion();

private:
    static std::unique_ptr<Inversion>& get_instance_ptr() {
        static std::unique_ptr<Inversion> instance;
        return instance;
    }

    void accumulate_smoothed_gradient(
        WaveType wt,
        int itype,
        real_t chi,
        const FieldVec &ker_smooth
    );
    real_t run_forward_adjoint(const bool is_calc_adj);
    void init_iteration();
    void steepest_descent();
    void grad_normalization(FieldVec &grads);
    void store_model();
    void store_gradient();
    void store_kernel_density();
    bool check_convergence();
    void model_update(FieldVec &dir);
    bool line_search();
    void write_src_rec_fwd();
    void write_obj_line();
    void alpha_clamp();

    std::ofstream obj_file_;  // objective function log; open only on main rank
    std::string   xdmf_fname_;

    Tensor3r model_update_;
    FieldVec ker_curr_, ker_prev_;
    FieldVec gradient_, gradient_prev_;

    std::vector<real_t> misfit_ = std::vector<real_t>(InputParams::IP().inversion().niter, _0_CR);
    real_t misfit_trial_ = _0_CR;
    int    iter_ = 0;
    int    iter_start_ = 0;
    real_t alpha_, alpha_R_, alpha_L_;
    bool   gradient_reuse_ = false;
    optimize::WolfeResult wolfe_res_ = {
        _0_CR, optimize::WolfeResult::Status::TRY
    };
    bool   break_flag_ = false;

    inline void convert_radial_kl() {
        gradient_[0] = gradient_[0] + gradient_[5]; // vs kernel
    }
};
