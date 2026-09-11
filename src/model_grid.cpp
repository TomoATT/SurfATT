#include "model_grid.h"
#include "src_rec.h"
#include "logger.h"
#include "inversion1d.h"
#include "h5io.h"
#include "utils.h"
#include "decomposer.h"

#include <algorithm>
#include <cmath>
#include <vector>


namespace {
    // Compute the bounding box (lon/lat min/max) of the station network
    // by scanning all station coordinates from the merged station table.
    void get_domain_min_max(real_t &lon_min, real_t &lon_max, real_t &lat_min, real_t &lat_max) {
        Eigen::VectorX<real_t> stla_vec = Eigen::Map<Eigen::VectorX<real_t>>(SrcRec::stas().stla.data(), SrcRec::stas().stla.size());
        Eigen::VectorX<real_t> stlo_vec = Eigen::Map<Eigen::VectorX<real_t>>(SrcRec::stas().stlo.data(), SrcRec::stas().stlo.size());

        lon_min = stlo_vec.minCoeff();
        lon_max = stlo_vec.maxCoeff();
        lat_min = stla_vec.minCoeff();
        lat_max = stla_vec.maxCoeff();
    }

    // Build depth anomalies as complete, symmetric half-sines. The top
    // layer has thickness asanom_size; the remaining layer thicknesses
    // follow an arithmetic progression that spans the full depth range.
    Eigen::VectorX<real_t> dep_anom(
        const Eigen::VectorX<real_t>& zgrids,
        const int nz,
        const real_t asanom_size
    ) {
        auto &logger = ATTLogger::logger();
        auto &mpi = Parallel::mpi();
        if (nz <= 0) {
            logger.Error("dep_anom: nz must be > 0", MODULE_GRID);
            mpi.abort(EXIT_FAILURE);
        }
        if (zgrids.size() < 2) {
            logger.Error("dep_anom: zgrids size must be >= 2", MODULE_GRID);
            mpi.abort(EXIT_FAILURE);
        }

        for (Eigen::Index k = 1; k < zgrids.size(); ++k) {
            if (zgrids(k) <= zgrids(k - 1)) {
                logger.Error(
                    "dep_anom: zgrids must be strictly increasing",
                    MODULE_GRID
                );
                mpi.abort(EXIT_FAILURE);
            }
        }

        const real_t z_top = zgrids(0);
        const real_t z_bottom = zgrids(zgrids.size() - 1);
        const real_t total_depth = z_bottom - z_top;

        Eigen::VectorX<real_t> boundaries(nz + 1);
        boundaries(0) = z_top;

        if (nz == 1) {
            boundaries(1) = z_bottom;
        } else {
            const real_t layer_size_inc =
                static_cast<real_t>(2)
                * (total_depth - static_cast<real_t>(nz) * asanom_size)
                / static_cast<real_t>(nz * (nz - 1));

            for (int layer = 0; layer < nz; ++layer) {
                const real_t layer_size =
                    asanom_size
                    + static_cast<real_t>(layer) * layer_size_inc;
                if (layer_size <= _0_CR) {
                    logger.Error(
                        fmt::format(
                            "dep_anom: invalid thickness {:.3f} km for "
                            "depth layer {}",
                            layer_size, layer + 1
                        ),
                        MODULE_GRID
                    );
                    mpi.abort(EXIT_FAILURE);
                }
                boundaries(layer + 1) =
                    boundaries(layer) + layer_size;
            }
        }

        // Remove accumulated round-off at the last layer boundary.
        boundaries(nz) = z_bottom;

        for (int layer = 0; layer < nz; ++layer) {
            logger.Debug(
                fmt::format(
                    "Depth anomaly layer {}: {:.3f}--{:.3f} km "
                    "(thickness {:.3f} km)",
                    layer + 1,
                    boundaries(layer),
                    boundaries(layer + 1),
                    boundaries(layer + 1) - boundaries(layer)
                ),
                MODULE_GRID
            );
        }

        Eigen::VectorX<real_t> pattern =
            Eigen::VectorX<real_t>::Zero(zgrids.size());
        int layer = 0;
        for (Eigen::Index k = 0; k < zgrids.size(); ++k) {
            const real_t depth = zgrids(k);
            while (
                layer < nz - 1
                && depth >= boundaries(layer + 1)
            ) {
                ++layer;
            }

            const real_t layer_size =
                boundaries(layer + 1) - boundaries(layer);
            const real_t local_depth = std::clamp(
                (depth - boundaries(layer)) / layer_size,
                _0_CR,
                _1_CR
            );
            const real_t sign =
                (layer % 2 == 0) ? _1_CR : -_1_CR;
            pattern(k) = sign * std::sin(PI * local_depth);
        }

        // Domain endpoints are layer boundaries and should be exactly zero.
        pattern(0) = _0_CR;
        pattern(pattern.size() - 1) = _0_CR;
        return pattern;
    }
}

// Constructor: derive grid dimensions from the station bounding box + margin,
// build evenly-spaced coordinate vectors, and initialise the 3-D model arrays.
ModelGrid::ModelGrid() {
    const auto &dom = InputParams::IP().domain();
    auto &logger = ATTLogger::logger();
    auto &mpi = Parallel::mpi();

    real_t xbeg, xend, ybeg, yend;
    if (dom.grid_method == 0) {
        // Grid spacing in lon, lat, and depth directions
        dgrid_i = dom.interval[0];
        dgrid_j = dom.interval[1];
        dgrid_k = dom.interval[2];

        // Expand the station bounding box by num_grid_margin cells on each side
        real_t lon_min, lon_max, lat_min, lat_max;
        get_domain_min_max(lon_min, lon_max, lat_min, lat_max);
        xbeg = lon_min - dom.num_grid_margin * dgrid_i;
        xend = lon_max + dom.num_grid_margin * dgrid_i;
        ybeg = lat_min - dom.num_grid_margin * dgrid_j;
        yend = lat_max + dom.num_grid_margin * dgrid_j;

        // Number of grid nodes (inclusive on both ends).
        // Round first to avoid floating-point drift (e.g. 99.9999999 -> 100).
        ngrid_i = static_cast<int>(std::llround((xend - xbeg) / dgrid_i) + 1);
        ngrid_j = static_cast<int>(std::llround((yend - ybeg) / dgrid_j) + 1);
        ngrid_k = static_cast<int>(std::llround((dom.depth[1] - dom.depth[0]) / dgrid_k) + 1);

    } else if (dom.grid_method == 1) {
        // Grid dimensions and bounding box are directly specified in the config
        ngrid_i = dom.n_grid[0];
        ngrid_j = dom.n_grid[1];
        ngrid_k = dom.n_grid[2];
        xbeg = dom.lon_min_max[0];
        xend = dom.lon_min_max[1];
        ybeg = dom.lat_min_max[0];
        yend = dom.lat_min_max[1];
        dgrid_i = (xend - xbeg) / static_cast<real_t>(ngrid_i - 1);
        dgrid_j = (yend - ybeg) / static_cast<real_t>(ngrid_j - 1);
        dgrid_k = (dom.depth[1] - dom.depth[0]) / static_cast<real_t>(ngrid_k - 1);
    } else {
        logger.Error(fmt::format("Unsupported grid_method {}", dom.grid_method), MODULE_GRID);
        mpi.abort(EXIT_FAILURE);
    }

    xgrids = Eigen::VectorX<real_t>::LinSpaced(ngrid_i, xbeg, xend);
    ygrids = Eigen::VectorX<real_t>::LinSpaced(ngrid_j, ybeg, yend);
    zgrids = Eigen::VectorX<real_t>::LinSpaced(ngrid_k, dom.depth[0], dom.depth[1]);

    logger.Info(
        fmt::format("Model grids: nx,ny,nz: {}, {}, {}", ngrid_i, ngrid_j, ngrid_k),
        MODULE_GRID
    );
    logger.Info(
        fmt::format("Longitude range: {:.3f} to {:.3f}", xbeg, xend),
        MODULE_GRID
    );
    logger.Info(
        fmt::format("Latitude range: {:.3f} to {:.3f}", ybeg, yend),
        MODULE_GRID
    );
    allocate_model_grids();

    Decomposer::DCP().subdomain_allocation(xgrids, ygrids);
}

void ModelGrid::allocate_model_grids() {
    auto &mpi = Parallel::mpi();
    auto &dcp = Decomposer::DCP();
    auto &IP = InputParams::IP();

    mpi.alloc_shared(ngrid_i * ngrid_j * ngrid_k, vp3d, win_vp_);
    mpi.alloc_shared(ngrid_i * ngrid_j * ngrid_k, vs3d, win_vs_);
    mpi.alloc_shared(ngrid_i * ngrid_j * ngrid_k, rho3d, win_rho_);
    if (IP.inversion().model_para_type == MODEL_AZI_ANI) {  // azimuthal anisotropy
        mpi.alloc_shared(ngrid_i * ngrid_j * ngrid_k, gc3d, win_gc_);
        mpi.alloc_shared(ngrid_i * ngrid_j * ngrid_k, gs3d, win_gs_);
    }
    if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
        mpi.alloc_shared(ngrid_i * ngrid_j * ngrid_k, vsh3d, win_vsh_);
    }
    // Always allocate local model slices; they are needed by fwdsurf() in both
    // FORWARD_ONLY and INVERSION_MODE.
    vs3d_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
    vp3d_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
    rho3d_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
    vs3d_loc.setZero();
    vp3d_loc.setZero();
    rho3d_loc.setZero();
    if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
        gc3d_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        gs3d_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        gc3d_loc.setZero();
        gs3d_loc.setZero();
    }
    if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
        vsh3d_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        gamma3d_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        vsh3d_loc.setZero();
        gamma3d_loc.setZero();
    }
}

// Build a linearly-interpolated 1-D Vs profile spanning [vel_range[0], vel_range[1]]
// at the same depth nodes as zgrids.
void ModelGrid::build_1d_model_linear() {
    vs1d = Eigen::VectorX<real_t>::LinSpaced(
        zgrids.size(), 
        InputParams::IP().model().vel_range[0], 
        InputParams::IP().model().vel_range[1]
    );
}

// Refine the linear starting model via 1-D surface-wave inversion.
// The linear profile is used as the initial guess.
void ModelGrid::build_1d_model_inversion() {
    build_1d_model_linear();
    const auto &IP = InputParams::IP();

    std::vector<Eigen::VectorX<real_t>> v1d_vec(2);  // 0: vsv, 1: vsh
    for (auto wt : {WaveType::RL, WaveType::LV}) {
        int iwt = static_cast<int>(wt);
        if (IP.data().wave_type[iwt]) {
            Inversion1D inv1d(wt);
            v1d_vec[iwt] = inv1d.inv1d(zgrids, vs1d);
            if (IP.inversion().model_para_type != MODEL_RADIAL_ANI) {
                vs1d = v1d_vec[iwt];  // update vs1d for the next inversion; final vs1d will be from the last inversion run
            }
        }
    }
    if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
        vs1d = vsvvsh2vs(v1d_vec[0], v1d_vec[1]);
    }
}

// Read vs (required) and optionally vp, rho, gc, gs from the HDF5 file.
// For any field absent in the file, empirical scaling is applied as fallback.
// Writes directly into the shared-memory arrays vp3d/vs3d/rho3d/gc3d/gs3d.
void ModelGrid::load_3d_model() {
    const auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();
    auto &mpi = Parallel::mpi();
    H5IO f(IP.model().init_model_path, H5IO::RDONLY);
    const hsize_t expect_n = static_cast<hsize_t>(ngrid_i * ngrid_j * ngrid_k);

    try {
        auto read_axis = [&](const std::string &name_new, const std::string &name_old) {
            std::vector<real_t> v;
            std::string used_name;
            if (f.exists(name_new)) {
                v = f.read_vector<real_t>(name_new);
                used_name = name_new;
            } else if (f.exists(name_old)) {
                v = f.read_vector<real_t>(name_old);
                used_name = name_old;
            } else {
                throw std::runtime_error(fmt::format(
                    "Missing coordinate axis '{}' (or legacy '{}') in initial model file",
                    name_new, name_old));
            }
            if (v.size() < 2) {
                throw std::runtime_error(fmt::format(
                    "Axis '{}' must contain at least 2 nodes, got {}", used_name, v.size()));
            }
            Eigen::VectorX<real_t> out = Eigen::Map<Eigen::VectorX<real_t>>(v.data(), static_cast<int>(v.size()));
            for (int i = 1; i < out.size(); ++i) {
                if (out(i) <= out(i - 1)) {
                    throw std::runtime_error(fmt::format(
                        "Axis '{}' must be strictly increasing (index {}: {} <= {})",
                        used_name, i, out(i), out(i - 1)));
                }
            }
            return out;
        };

        const Eigen::VectorX<real_t> xsrc = read_axis("x", "xgrids");
        const Eigen::VectorX<real_t> ysrc = read_axis("y", "ygrids");
        const Eigen::VectorX<real_t> zsrc = read_axis("z", "zgrids");

        auto axis_equal = [](const Eigen::VectorX<real_t>& a, const Eigen::VectorX<real_t>& b) {
            if (a.size() != b.size()) return false;
            for (int i = 0; i < a.size(); ++i) {
                if (!real_t_equal(a(i), b(i))) return false;
            }
            return true;
        };

        const bool same_grid =
            axis_equal(xsrc, xgrids) &&
            axis_equal(ysrc, ygrids) &&
            axis_equal(zsrc, zgrids);

        auto check_target_inside_source = [&](const Eigen::VectorX<real_t>& src,
                                              const Eigen::VectorX<real_t>& dst,
                                              const std::string& axis_name) {
            const real_t s0 = src(0);
            const real_t s1 = src(src.size() - 1);
            const real_t d0 = dst(0);
            const real_t d1 = dst(dst.size() - 1);
            const real_t tol = static_cast<real_t>(1.0e-6) * std::max(static_cast<real_t>(1), std::abs(s1 - s0));
            if (d0 < s0 - tol || d1 > s1 + tol) {
                throw std::runtime_error(fmt::format(
                    "Target {} grid [{:.6f}, {:.6f}] is outside source grid [{:.6f}, {:.6f}]",
                    axis_name, d0, d1, s0, s1));
            }
        };

        if (!same_grid) {
            check_target_inside_source(xsrc, xgrids, "x");
            check_target_inside_source(ysrc, ygrids, "y");
            check_target_inside_source(zsrc, zgrids, "z");
        }

        auto interpolate_or_copy = [&](const std::string &name, real_t *dst) {
            hsize_t nx = 0, ny = 0, nz = 0;
            auto src = f.read_volume<real_t>(name, nx, ny, nz);
            if (nx != static_cast<hsize_t>(xsrc.size()) ||
                ny != static_cast<hsize_t>(ysrc.size()) ||
                nz != static_cast<hsize_t>(zsrc.size())) {
                throw std::runtime_error(fmt::format(
                    "Dataset '{}' shape ({},{},{}) is inconsistent with source grids ({},{},{})",
                    name, nx, ny, nz, xsrc.size(), ysrc.size(), zsrc.size()));
            }

            if (same_grid &&
                nx == static_cast<hsize_t>(ngrid_i) &&
                ny == static_cast<hsize_t>(ngrid_j) &&
                nz == static_cast<hsize_t>(ngrid_k)) {
                std::copy(src.begin(), src.end(), dst);
                return;
            }

            const real_t x0 = xsrc(0), x1 = xsrc(xsrc.size() - 1);
            const real_t y0 = ysrc(0), y1 = ysrc(ysrc.size() - 1);
            const real_t z0 = zsrc(0), z1 = zsrc(zsrc.size() - 1);

            for (int i = 0; i < ngrid_i; ++i) {
                real_t qx = xgrids(i);
                if (qx < x0) qx = x0;
                if (qx > x1) qx = x1;
                for (int j = 0; j < ngrid_j; ++j) {
                    real_t qy = ygrids(j);
                    if (qy < y0) qy = y0;
                    if (qy > y1) qy = y1;
                    for (int k = 0; k < ngrid_k; ++k) {
                        real_t qz = zgrids(k);
                        if (qz < z0) qz = z0;
                        if (qz > z1) qz = z1;
                        const real_t val = trilinear_interpolation(
                            xsrc.data(), ysrc.data(), zsrc.data(),
                            static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz),
                            src.data(),
                            qx, qy, qz
                        );
                        if (std::isnan(val)) {
                            throw std::runtime_error(fmt::format(
                                "Trilinear interpolation failed for '{}' at ({:.6f},{:.6f},{:.6f})",
                                name, qx, qy, qz));
                        }
                        dst[I2V(i, j, k)] = val;
                    }
                }
            }
        };

        // --- vs or vsv (required) ---
        {
            std::string key = IP.inversion().model_para_type == MODEL_RADIAL_ANI ? "vsv" : "vs";
            interpolate_or_copy(key, vs3d);
            logger.Info(
                same_grid ? "Loaded '" + key + "' from HDF5 file." :
                            "Loaded and interpolated '" + key + "' onto current model grid.",
                MODULE_GRID
            );
        }

        // --- vp (optional, fallback: empirical vs2vp) ---
        if (f.exists("vp")) {
            interpolate_or_copy("vp", vp3d);
            logger.Info(
                same_grid ? "Loaded 'vp' from HDF5 file." :
                            "Loaded and interpolated 'vp' onto current model grid.",
                MODULE_GRID
            );
        } else {
            logger.Info("'vp' not found in HDF5 file, computing from empirical vs2vp.", MODULE_GRID);
            for (hsize_t i = 0; i < expect_n; ++i)
                vp3d[i] = vs2vp(vs3d[i]);
        }

        // --- rho (optional, fallback: empirical vp2rho) ---
        if (f.exists("rho")) {
            interpolate_or_copy("rho", rho3d);
            logger.Info(
                same_grid ? "Loaded 'rho' from HDF5 file." :
                            "Loaded and interpolated 'rho' onto current model grid.",
                MODULE_GRID
            );
        } else {
            logger.Info("'rho' not found in HDF5 file, computing from empirical vp2rho.", MODULE_GRID);
            for (hsize_t i = 0; i < expect_n; ++i)
                rho3d[i] = vp2rho(vp3d[i]);
        }

        // --- gc / gs (optional, only if model_para_type is MODEL_AZI_ANI) ---
        if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
            if (f.exists("gc")) {
                interpolate_or_copy("gc", gc3d);
                logger.Info(
                    same_grid ? "Loaded 'gc' from HDF5 file." :
                                "Loaded and interpolated 'gc' onto current model grid.",
                    MODULE_GRID
                );
            } else {
                logger.Info("'gc' not found in HDF5 file, initialising to zero.", MODULE_GRID);
                std::fill(gc3d, gc3d + expect_n, _0_CR);
            }
            if (f.exists("gs")) {
                interpolate_or_copy("gs", gs3d);
                logger.Info(
                    same_grid ? "Loaded 'gs' from HDF5 file." :
                                "Loaded and interpolated 'gs' onto current model grid.",
                    MODULE_GRID
                );
            } else {
                logger.Info("'gs' not found in HDF5 file, initialising to zero.", MODULE_GRID);
                std::fill(gs3d, gs3d + expect_n, _0_CR);
            }
        }
        
        // --- vsh (optional, only if model_para_type is MODEL_RADIAL_ANI) ---
        if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
            if (f.exists("vsh")) {
                interpolate_or_copy("vsh", vsh3d);
                logger.Info(
                    same_grid ? "Loaded 'vsh' from HDF5 file." :
                                "Loaded and interpolated 'vsh' onto current model grid.",
                    MODULE_GRID
                );
            } else {
                logger.Info("'vsh' not found in HDF5 file, initialising to vs (isotropic).", MODULE_GRID);
                for (hsize_t i = 0; i < expect_n; ++i)
                    vsh3d[i] = vs3d[i];
            }
            for (hsize_t i = 0; i < expect_n; ++i){
                if (vsh3d[i] == 0) {
                    logger.Error(fmt::format(
                        "Invalid 'vsh' value of zero at index {} in HDF5 file, which would cause division by zero when computing gamma. Please fix the input model file.",
                        i
                    ), MODULE_GRID);
                    mpi.abort(EXIT_FAILURE);
                }
            }
        }
    } catch (const std::exception &e) {
        logger.Error(fmt::format(
            "ModelGrid: failed to load 3D model from HDF5 file '{}': {}",
            IP.model().init_model_path, e.what()
        ), MODULE_GRID);
        mpi.abort(EXIT_FAILURE);
    }

    // Compute depth-averaged vs1d from the loaded vs3d
    vs1d.setZero(ngrid_k);
    for (int k = 0; k < ngrid_k; ++k) {
        for (int j = 0; j < ngrid_j; ++j)
            for (int i = 0; i < ngrid_i; ++i)
                vs1d(k) += vs3d[I2V(i, j, k)];
        vs1d(k) /= (ngrid_i * ngrid_j);
    }
}

// Allocate MPI shared-memory windows for vp3d/vs3d/rho3d, populate the arrays
// according to init_model_type, then broadcast to all ranks.
//
// init_model_type:
//   0 — linear 1-D profile extruded into 3-D
//   1 — 1-D surface-wave inversion result extruded into 3-D
//   2 — full 3-D model loaded directly from an HDF5 file
void ModelGrid::build_init_model() {
    const auto &IP = InputParams::IP();
    auto &mpi = Parallel::mpi();
    auto &logger = ATTLogger::logger();

    if (IP.model().init_model_type == 0) {
        // Build a simple linear Vs gradient
        logger.Info(fmt::format(
            "Building linear 1-D initial model with S-wave velocity from {:.2f} to {:.2f} km/s", 
            IP.model().vel_range[0], IP.model().vel_range[1]), MODULE_GRID
        );
        build_1d_model_linear();
    } else if (IP.model().init_model_type == 1) {
        // Invert average dispersion curves to obtain a 1-D Vs model
        logger.Info(
            "Building 1-D initial model from surface-wave inversion of average dispersion curves",
            MODULE_GRID
        );
        build_1d_model_inversion();
    } else if (IP.model().init_model_type == 2) {
        // Load an externally-supplied 3-D model; only main rank does the I/O
        logger.Info(fmt::format(
            "Building 3-D initial model from HDF5 file: {}",
            IP.model().init_model_path),
            MODULE_GRID
        );
        if (mpi.is_main()) {
            load_3d_model();  // fills vs3d/vp3d/rho3d and optionally gc3d/gs3d
        }
        mpi.barrier();
    } else {
        logger.Error(fmt::format("Unsupported init_model_type {}", IP.model().init_model_type), MODULE_GRID);
        mpi.abort(EXIT_FAILURE);
    }

    if (IP.model().init_model_type != 2) {
        // Extrude the 1-D profile laterally to fill the full 3-D volume
        if (mpi.is_main()) {
            for (int ix = 0; ix < ngrid_i; ++ix) {
                for (int iy = 0; iy < ngrid_j; ++iy) {
                    for (int iz = 0; iz < ngrid_k; ++iz) {
                        vp3d[I2V(ix, iy, iz)] = vs2vp(vs1d(iz));
                        vs3d[I2V(ix, iy, iz)] = vs1d(iz);
                        rho3d[I2V(ix, iy, iz)] = vp2rho(vp3d[I2V(ix, iy, iz)]);  // empirical Vp→density scaling
                        if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
                            gc3d[I2V(ix, iy, iz)] = _0_CR;  // start with isotropic model
                            gs3d[I2V(ix, iy, iz)] = _0_CR;  // start with isotropic model
                        }
                        if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
                            vsh3d[I2V(ix, iy, iz)] = vs1d(iz);  // start with isotropic model
                        }
                    }
                }
            }
        }
        // write() guards itself with is_main() and calls mpi.barrier() internally,
        // so it must be called by all ranks — not inside if (mpi.is_main()).
        mpi.barrier();
    }
    if (IP.output().output_initial_model) {
        write(INIT_MODEL_FNAME);
    }

    // Broadcast the model from main rank to all other ranks.
    // For init_model_type 2 only the main rank has vs1d populated, so all
    // non-main ranks must resize it before the bcast to avoid a null-ptr crash.
    if (!mpi.is_main()) vs1d.resize(ngrid_k);
    mpi.bcast(vs1d.data(), ngrid_k);
    mpi.sync_from_main_rank(vp3d,  ngrid_i * ngrid_j * ngrid_k);
    mpi.sync_from_main_rank(vs3d,  ngrid_i * ngrid_j * ngrid_k);
    mpi.sync_from_main_rank(rho3d, ngrid_i * ngrid_j * ngrid_k);
    if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
        mpi.sync_from_main_rank(gc3d, ngrid_i * ngrid_j * ngrid_k);
        mpi.sync_from_main_rank(gs3d, ngrid_i * ngrid_j * ngrid_k);
    }
    if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
        mpi.sync_from_main_rank(vsh3d, ngrid_i * ngrid_j * ngrid_k);
    }
}

std::tuple<Eigen::VectorX<real_t>, Eigen::VectorX<real_t>, Eigen::VectorX<real_t>>
ModelGrid::build_perturbation_pattern(
    int nx_w, int ny_w, int nz_w, real_t hmargin, real_t anom_sz
) const {
    auto &logger = ATTLogger::logger();
    auto &mpi = Parallel::mpi();
    const int ntaper_i = static_cast<int>(hmargin / dgrid_i);
    const int ntaper_j = static_cast<int>(hmargin / dgrid_j);
    const int inner_i  = ngrid_i - 2 * ntaper_i;
    const int inner_j  = ngrid_j - 2 * ntaper_j;

    if (inner_i <= 0 || inner_j <= 0){
        logger.Error("ModelGrid::build_perturbation_pattern: hmargin too large for current grid size", MODULE_GRID);
        mpi.abort(EXIT_FAILURE);
    }

    Eigen::VectorX<real_t> xp = Eigen::VectorX<real_t>::Zero(ngrid_i);
    Eigen::VectorX<real_t> yp = Eigen::VectorX<real_t>::Zero(ngrid_j);
    Eigen::VectorX<real_t> zp = Eigen::VectorX<real_t>::Zero(ngrid_k);

    {
        Eigen::VectorX<real_t> ii = Eigen::VectorX<real_t>::LinSpaced(
            inner_i, _1_CR, static_cast<real_t>(inner_i));
        xp.segment(ntaper_i, inner_i) =
            (static_cast<real_t>(nx_w) * PI * ii.array() / static_cast<real_t>(inner_i)).sin().matrix();
    }
    {
        Eigen::VectorX<real_t> jj = Eigen::VectorX<real_t>::LinSpaced(
            inner_j, _1_CR, static_cast<real_t>(inner_j));
        yp.segment(ntaper_j, inner_j) =
            (static_cast<real_t>(ny_w) * PI * jj.array() / static_cast<real_t>(inner_j)).sin().matrix();
    }
    if (anom_sz <= _0_CR) {
        Eigen::VectorX<real_t> kk = Eigen::VectorX<real_t>::LinSpaced(
            ngrid_k, _1_CR, static_cast<real_t>(ngrid_k));
        zp = (static_cast<real_t>(nz_w) * PI * kk.array() / static_cast<real_t>(ngrid_k)).sin().matrix();
    } else {
        zp = dep_anom(zgrids, nz_w, anom_sz);
    }
    return {xp, yp, zp};
}

void ModelGrid::add_perturbation(
    const int nx, const int ny, const int nz,
    const real_t pert_vel, const real_t hmargin,
    const real_t anom_size, const bool only_vs
) {
    auto &mpi = Parallel::mpi();
    const int nelem = ngrid_i * ngrid_j * ngrid_k;

    if (mpi.is_main()) {
        auto [x_pert, y_pert, z_pert] = build_perturbation_pattern(nx, ny, nz, hmargin, anom_size);
        for (int i = 0; i < ngrid_i; ++i) {
            for (int j = 0; j < ngrid_j; ++j) {
                const real_t amp = x_pert(i) * y_pert(j) * pert_vel;
                Eigen::Map<Eigen::VectorX<real_t>> vs_line(&vs3d[I2V(i, j, 0)], ngrid_k);
                vs_line.array() *= (_1_CR + amp * z_pert.array());
            }
        }
        if (!only_vs) {
            Eigen::Map<Eigen::VectorX<real_t>> vs_vec(vs3d, nelem);
            Eigen::Map<Eigen::VectorX<real_t>> vp_vec(vp3d, nelem);
            Eigen::Map<Eigen::VectorX<real_t>> rho_vec(rho3d, nelem);
            vp_vec = vs2vp<real_t>(vs_vec);
            rho_vec = vp2rho<real_t>(vp_vec);
        }
    }

    mpi.barrier();
    mpi.sync_from_main_rank(vs3d, nelem);
    mpi.sync_from_main_rank(vp3d, nelem);
    mpi.sync_from_main_rank(rho3d, nelem);
}

void ModelGrid::add_azi_aniso_perturbation(
    const int nx, const int ny, const int nz,
    const real_t angle, const real_t pert_ani,
    const real_t hmargin, const real_t anom_size
) {
    auto &mpi = Parallel::mpi();
    auto &IP = InputParams::IP();
    const int nelem = ngrid_i * ngrid_j * ngrid_k;

    if (IP.inversion().model_para_type != MODEL_AZI_ANI) return;

    if (mpi.is_main()) {
        auto [xp, yp, zp] = build_perturbation_pattern(nx, ny, nz, hmargin, anom_size);
        for (int i = 0; i < ngrid_i; ++i) {
            for (int j = 0; j < ngrid_j; ++j) {
                for (int k = 0; k < ngrid_k; ++k) {
                    const real_t amp = xp(i) * yp(j) * zp(k) * pert_ani;
                    if (amp > _0_CR) {
                        gc3d[I2V(i, j, k)] = std::abs(amp) * std::cos(2 * angle * DEG2RAD);
                        gs3d[I2V(i, j, k)] = std::abs(amp) * std::sin(2 * angle * DEG2RAD);
                    } else {
                        gc3d[I2V(i, j, k)] = std::abs(amp) * std::cos(2 * (angle + 90) * DEG2RAD);
                        gs3d[I2V(i, j, k)] = std::abs(amp) * std::sin(2 * (angle + 90) * DEG2RAD);
                    }
                }
            }
        }
    }

    mpi.barrier();
    mpi.sync_from_main_rank(gc3d, nelem);
    mpi.sync_from_main_rank(gs3d, nelem);
}

void ModelGrid::add_radial_aniso_perturbation(
    const int nx, const int ny, const int nz,
    const real_t pert_vs, const real_t pert_zeta,
    const real_t hmargin, const real_t anom_size
) {
    auto &mpi = Parallel::mpi();
    auto &IP = InputParams::IP();
    const int nelem = ngrid_i * ngrid_j * ngrid_k;

    if (IP.inversion().model_para_type != MODEL_RADIAL_ANI) return;

    if (mpi.is_main()) {
        auto [xp, yp, zp] = build_perturbation_pattern(nx, ny, nz, hmargin, anom_size);
        for (int i = 0; i < ngrid_i; ++i) {
            for (int j = 0; j < ngrid_j; ++j) {
                for (int k = 0; k < ngrid_k; ++k) {
                    const real_t amp_vs = xp(i) * yp(j) * zp(k) * pert_vs;
                    const real_t amp_zeta = xp(i) * yp(j) * zp(k) * pert_zeta;

                    const int idx = I2V(i, j, k);

                    // Calculate original Vs and zeta
                    // Vs = sqrt((2*vsv^2 + vsh^2)/3), zeta = vsh^2/vsv^2
                    const real_t Vs0 = vs3d[idx];
                    const real_t zeta0 = _1_CR;

                    // Apply perturbations to Vs and zeta
                    const real_t Vs_new = Vs0 * (1 + amp_vs);
                    const real_t zeta_new = zeta0 * (1 + amp_zeta);

                    // Recover vsv and vsh from Vs and zeta
                    // vsv = sqrt(3*Vs^2 / (2 + zeta))
                    // vsh = vsv * sqrt(zeta)
                    auto [vsv_new, vsh_new] = recover_anisotropy(Vs_new, zeta_new);

                    // Update vs3d to store Vs (for consistency with load_3d_model)
                    vs3d[idx] = vsv_new;
                    vsh3d[idx] = vsh_new;
                    vp3d[idx] = vs2vp(vsv_new);
                    rho3d[idx] = vp2rho(vp3d[idx]);
                }
            }
        }
    }

    mpi.barrier();
    mpi.sync_from_main_rank(vs3d, nelem);
    mpi.sync_from_main_rank(vsh3d, nelem);
    mpi.sync_from_main_rank(vp3d, nelem);
    mpi.sync_from_main_rank(rho3d, nelem);
}

void ModelGrid::collect_model_loc() {
    auto &mpi = Parallel::mpi();
    auto &dcp = Decomposer::DCP();
    auto &IP = InputParams::IP();

    int nelem = ngrid_i * ngrid_j * ngrid_k;
    Tensor3r vs_tmp = dcp.collect_data(vs3d_loc.data());
    Tensor3r vp_tmp = dcp.collect_data(vp3d_loc.data());
    Tensor3r rho_tmp = dcp.collect_data(rho3d_loc.data());
    Tensor3r gc_tmp, gs_tmp, vsh_tmp;
    if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
        gc_tmp = dcp.collect_data(gc3d_loc.data());
        gs_tmp = dcp.collect_data(gs3d_loc.data());
    }
    if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
        vsh_tmp = dcp.collect_data(vsh3d_loc.data());
    }
    if (mpi.is_main()) {
        std::copy(vs_tmp.data(), vs_tmp.data() + nelem, vs3d);
        std::copy(vp_tmp.data(), vp_tmp.data() + nelem, vp3d);
        std::copy(rho_tmp.data(), rho_tmp.data() + nelem, rho3d);
         if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
            std::copy(gc_tmp.data(), gc_tmp.data() + nelem, gc3d);
            std::copy(gs_tmp.data(), gs_tmp.data() + nelem, gs3d);
        }
        if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
            std::copy(vsh_tmp.data(), vsh_tmp.data() + nelem, vsh3d);
        }
    }
    mpi.barrier();
    mpi.sync_from_main_rank(vs3d, nelem);
    mpi.sync_from_main_rank(vp3d, nelem);
    mpi.sync_from_main_rank(rho3d, nelem);
    if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
        mpi.sync_from_main_rank(gc3d, nelem);
        mpi.sync_from_main_rank(gs3d, nelem);
    }
    if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
        mpi.sync_from_main_rank(vsh3d, nelem);
    }
   
}

void ModelGrid::write(const std::string &subname) {
    auto &IP = InputParams::IP();
    auto &mpi = Parallel::mpi();
    if (mpi.is_main()) {
        std::string filename = fmt::format("{}/{}", IP.output().output_path, subname);
        H5IO f(filename, H5IO::TRUNC);
        f.write_vector("x", xgrids);
        f.write_vector("y", ygrids);
        f.write_vector("z", zgrids);

        const int nelem = ngrid_i * ngrid_j * ngrid_k;

        if (IP.inversion().model_para_type == MODEL_RADIAL_ANI) {
            // For radial anisotropy: vs3d stores vsv, vsh3d stores vsh
            // Write vsv and vsh
            f.write_volume("vsv", vs3d, ngrid_i, ngrid_j, ngrid_k);
            f.write_volume("vsh", vsh3d, ngrid_i, ngrid_j, ngrid_k);

            // Compute and write Vs = sqrt((2*vsv + vsh)/3) and zeta = vsh^2/vsv^2
            std::vector<real_t> Vs(nelem);
            std::vector<real_t> zeta(nelem);
            for (int i = 0; i < nelem; ++i) {
                Vs[i] = vsvvsh2vs(vs3d[i], vsh3d[i]);
                zeta[i] = vsvvsh2zeta(vs3d[i], vsh3d[i]);
            }
            f.write_volume("vs", Vs, ngrid_i, ngrid_j, ngrid_k);
            f.write_volume("zeta", zeta, ngrid_i, ngrid_j, ngrid_k);

            // Write vp and rho if available
            if (IP.inversion().use_alpha_beta_rho) {
                f.write_volume("vp", vp3d, ngrid_i, ngrid_j, ngrid_k);
                f.write_volume("rho", rho3d, ngrid_i, ngrid_j, ngrid_k);
            }
        } else {
            // Isotropic or azimuthal anisotropy case
            f.write_volume("vs", vs3d, ngrid_i, ngrid_j, ngrid_k);
            if (IP.inversion().use_alpha_beta_rho) {
                f.write_volume("vp", vp3d, ngrid_i, ngrid_j, ngrid_k);
                f.write_volume("rho", rho3d, ngrid_i, ngrid_j, ngrid_k);
            }
            if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
                f.write_volume("gc", gc3d, ngrid_i, ngrid_j, ngrid_k);
                f.write_volume("gs", gs3d, ngrid_i, ngrid_j, ngrid_k);
                std::vector<real_t> g0(nelem);
                std::vector<real_t> theta(nelem);
                for (int i = 0; i < nelem; ++i) {
                    g0[i] = std::sqrt(gc3d[i] * gc3d[i] + gs3d[i] * gs3d[i]);
                    theta[i] = _0_5_CR * std::atan2(gs3d[i], gc3d[i]) * RAD2DEG;
                }
                f.write_volume("g0", g0, ngrid_i, ngrid_j, ngrid_k);
                f.write_volume("theta", theta, ngrid_i, ngrid_j, ngrid_k);
            }
        }
    }
    mpi.barrier();
}

// Free the MPI shared-memory windows and null the associated raw pointers.
void ModelGrid::release_shm(){
    auto &mpi = Parallel::mpi();

    mpi.free_shared(vp3d, win_vp_);
    mpi.free_shared(vs3d, win_vs_);
    mpi.free_shared(rho3d, win_rho_);
    mpi.free_shared(gc3d, win_gc_);
    mpi.free_shared(gs3d, win_gs_);
}
