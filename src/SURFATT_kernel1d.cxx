// SURFATT_kernel1d
//
// Compute surface-wave dispersion and 1-D depth sensitivity kernels for a
// single 1-D Vs profile with the same surfker engine used by SURFATT_tomo
// (same layering, Earth flattening, fundamental mode and Brocher vp/rho).
//
// Input HDF5 datasets (1-D):
//   z        depth nodes (km), strictly increasing, shape (nz)
//   vs       Vs at the depth nodes (km/s), > 0, shape (nz)
//   periods  periods (s), > 0 and strictly increasing, shape (nper)
//
// Output HDF5 datasets:
//   z, periods
//   vel           phase or group velocity (km/s), shape (nper)
//   vp, rho       empirical Vp (km/s) and density (g/cm^3), shape (nz)
//   sen_vs        d(vel)/d(vs),  shape (nper, nz)
//   sen_vp        d(vel)/d(vp),  shape (nper, nz); zeros for Love waves
//   sen_rho       d(vel)/d(rho), shape (nper, nz)
//   sen_vs_total  vs kernel as combined by the vs-only inversion of SURFATT_tomo,
//                 shape (nper, nz): for Rayleigh waves vp and rho are scaled
//                 from vs by the empirical relations; for Love waves it equals
//                 sen_vs
//
// A period without a root, or any NaN/Inf in the velocities or kernels, is
// reported as an error.

#include "surfker/surfker.hpp"
#include "h5io.h"
#include "argparser.h"
#include "logger.h"
#include "parallel.h"
#include "utils.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Kernel1DArgs {
    std::string fname;
    std::string outfname;
    WaveType wave_type = WaveType::RL;
    SurfType surf_type = SurfType::PH;
};

Kernel1DArgs argparse_kernel1d(int argc, char* argv[]) {
    ArgList al(argc, argv);
    if (al.empty() || al.has("-h")) {
        std::cout <<
            "Usage: SURFATT_kernel1d -i model_file -o out_file [-w RL|LV] [-t PH|GR] [-h]\n\n"
            "Compute dispersion and 1-D depth sensitivity kernels of a 1-D Vs model\n"
            "with the same surfker engine used by SURFATT_tomo.\n\n"
            "required arguments:\n"
            "  -i model_file        HDF5 file with datasets z (km), vs (km/s) and periods (s)\n"
            "  -o out_file          Output HDF5 file with vel, sen_vs, sen_vp, sen_rho\n"
            "                       and sen_vs_total\n\n"
            "optional arguments:\n"
            "  -w RL|LV             Wave type: Rayleigh or Love (default: RL)\n"
            "  -t PH|GR             Velocity type: phase or group (default: PH)\n"
            "  -h                   Print help message\n";
        std::exit(0);
    }
    Kernel1DArgs out;
    out.fname    = al.require("-i");
    out.outfname = al.require("-o");
    if (auto v = al.get("-w")) {
        if (*v == "RL")      out.wave_type = WaveType::RL;
        else if (*v == "LV") out.wave_type = WaveType::LV;
        else throw std::runtime_error("-w must be RL or LV, got \"" + *v + "\"");
    }
    if (auto v = al.get("-t")) {
        if (*v == "PH")      out.surf_type = SurfType::PH;
        else if (*v == "GR") out.surf_type = SurfType::GR;
        else throw std::runtime_error("-t must be PH or GR, got \"" + *v + "\"");
    }
    return out;
}

Eigen::VectorX<real_t> read_eigen_vector(const H5IO &file, const std::string &name) {
    if (!file.exists(name)) {
        throw std::runtime_error("Required dataset '" + name + "' was not found");
    }
    auto v = file.read_vector<real_t>(name);
    return Eigen::Map<Eigen::VectorX<real_t>>(v.data(), static_cast<Eigen::Index>(v.size()));
}

// SURFATT_tomo always uses a strictly increasing depth grid and a sorted,
// unique period list; surfker relies on both (layer thicknesses from z,
// root bracketing from the previous period).
void require_strictly_increasing(const Eigen::VectorX<real_t> &v, const std::string &name) {
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v(i))) {
            throw std::runtime_error(fmt::format("{} has a non-finite value at index {}", name, i));
        }
        if (i > 0 && v(i) <= v(i - 1)) {
            throw std::runtime_error(fmt::format(
                "{} must be strictly increasing (index {}: {} <= {})", name, i, v(i), v(i - 1)));
        }
    }
}

// Throw if any entry of a kernel matrix is NaN/Inf.
void require_finite(const Eigen::MatrixX<real_t> &M, const std::string &name,
                    const Eigen::VectorX<real_t> &periods) {
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        if (!M.row(i).allFinite()) {
            throw std::runtime_error(fmt::format(
                "NaN/Inf in {} at period {:.3f} s", name, periods(i)));
        }
    }
}

int kernel1d(const Kernel1DArgs &args, ATTLogger &logger) {
    const std::string tag = waveTypeStr[static_cast<int>(args.wave_type)] + "_" +
                            surfTypeStr[static_cast<int>(args.surf_type)];

    // read 1-D model
    H5IO fin(args.fname, H5IO::RDONLY);
    const Eigen::VectorX<real_t> z = read_eigen_vector(fin, "z");
    const Eigen::VectorX<real_t> vs = read_eigen_vector(fin, "vs");
    const Eigen::VectorX<real_t> periods = read_eigen_vector(fin, "periods");
    const int nz = static_cast<int>(z.size());
    const int nper = static_cast<int>(periods.size());
    if (vs.size() != nz || nz < 2) {
        throw std::runtime_error("z and vs must have the same length (>= 2)");
    }
    if (nper < 1) {
        throw std::runtime_error("periods must not be empty");
    }
    require_strictly_increasing(z, "z");
    require_strictly_increasing(periods, "periods");
    if (periods(0) <= _0_CR) {
        throw std::runtime_error("periods must be positive");
    }
    if (!vs.allFinite() || (vs.array() <= _0_CR).any()) {
        throw std::runtime_error("vs must be finite and positive");
    }
    logger.Info(fmt::format("Computing {} dispersion and kernels: {} depth nodes, {} periods",
                            tag, nz, nper), MODULE_MAIN);

    // dispersion and kernels, vp and rho from empirical relations
    auto req = surfker::build_disp_req(z, vs, periods, IFLSPH, iwave_of(args.wave_type),
                                       IMODE, static_cast<int>(args.surf_type));
    const Eigen::VectorX<real_t> vel = surfker::surfdisp(req);
    for (int i = 0; i < nper; ++i) {
        if (!std::isfinite(vel(i))) {
            throw std::runtime_error(fmt::format(
                "NaN/Inf {} velocity at period {:.3f} s", tag, periods(i)));
        }
        // surfker zero-fills vel from the first failed period on, and its
        // kernels there are NaN
        if (vel(i) <= _0_CR) {
            throw std::runtime_error(fmt::format(
                "No {} root found in the fundamental mode at period {:.3f} s", tag, periods(i)));
        }
    }
    surfker::DepthKernel1D K = surfker::depthkernel1d(req);

    // Love-wave kernels have no vp sensitivity
    const auto full_or_zero = [&](const Eigen::MatrixX<real_t> &M) {
        if (M.rows() == nper && M.cols() == nz) return Eigen::MatrixX<real_t>(M);
        return Eigen::MatrixX<real_t>(Eigen::MatrixX<real_t>::Zero(nper, nz));
    };
    const Eigen::MatrixX<real_t> sen_vs = full_or_zero(K.sen_vs);
    const Eigen::MatrixX<real_t> sen_vp = full_or_zero(K.sen_vp);
    const Eigen::MatrixX<real_t> sen_rho = full_or_zero(K.sen_rho);

    // vs kernel as combined in preproc::combine_kernels (vs-only parametrisation):
    //   Rayleigh: K_vs + K_vp * d(vp)/d(vs) + K_rho * d(rho)/d(vp) * d(vp)/d(vs)
    //   Love:     K_vs (SURFATT_tomo keeps no vp or rho kernel for Love waves)
    Eigen::MatrixX<real_t> sen_vs_total = sen_vs;
    if (args.wave_type == WaveType::RL) {
        for (int k = 0; k < nz; ++k) {
            const real_t dab = dalpha_dbeta(req.vs_km_s(k));
            const real_t dra = drho_dalpha(req.vp_km_s(k));
            sen_vs_total.col(k) += sen_vp.col(k) * dab + sen_rho.col(k) * dra * dab;
        }
    }
    require_finite(sen_vs, "sen_vs", periods);
    require_finite(sen_vp, "sen_vp", periods);
    require_finite(sen_rho, "sen_rho", periods);

    // write results
    H5IO fout(args.outfname, H5IO::TRUNC);
    fout.write_vector("z", z);
    fout.write_vector("periods", periods);
    fout.write_vector("vel", vel);
    fout.write_vector("vp", req.vp_km_s.head(nz));
    fout.write_vector("rho", req.rho_g_cm3.head(nz));
    fout.write_matrix("sen_vs", sen_vs);
    fout.write_matrix("sen_vp", sen_vp);
    fout.write_matrix("sen_rho", sen_rho);
    fout.write_matrix("sen_vs_total", sen_vs_total);
    logger.Info(fmt::format("Kernels written to {}", args.outfname), MODULE_MAIN);
    return EXIT_SUCCESS;
}

} // namespace


int main(int argc, char* argv[]) {
    auto args = argparse_kernel1d(argc, argv);

    // initialise MPI
    Parallel::init();
    auto &mpi = Parallel::mpi();

    // logger
    ATTLogger::init("", /*log_level=*/2, /*console_only=*/true);
    auto &logger = ATTLogger::logger();

    if (mpi.size() > 1) {
        logger.Error("SURFATT_kernel1d is not designed for parallel execution. Please run with a single process.", MODULE_MAIN);
        mpi.finalize();
        return EXIT_FAILURE;
    }

    // Prevent the HDF5 library from printing its own diagnostic stack.  Errors
    // are caught below and reported once through the application logger.
    H5::Exception::dontPrint();
    int status = EXIT_FAILURE;
    try {
        status = kernel1d(args, logger);
    } catch (const H5::Exception &e) {
        logger.Error(fmt::format(
            "HDF5 error while processing '{}' or '{}': {}",
            args.fname, args.outfname, e.getDetailMsg()), MODULE_MAIN);
    } catch (const std::exception &e) {
        logger.Error(fmt::format(
            "Failed to compute 1-D kernels for '{}': {}", args.fname, e.what()), MODULE_MAIN);
    } catch (...) {
        logger.Error(fmt::format(
            "Unknown error while processing '{}'", args.fname), MODULE_MAIN);
    }

    mpi.finalize();
    return status;
}
