#pragma once

#include "config.h"
#include "rapidcsv.h"
#include "input_params.h"
#include "logger.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <mpi.h>
#include <map>
#include <string>
#include <vector>

struct event_info {
    real_t evla;
    real_t evlo;
    real_t period;
    int iper;
    std::vector<int> rec_indices;
    Eigen::VectorX<real_t> syn_data;
};

struct Stations {
    std::vector<std::string> stnm;
    std::vector<real_t> stla;
    std::vector<real_t> stlo;
};

struct PeriodInfo {
    Eigen::VectorX<real_t> periods;
    Eigen::VectorX<real_t> meanvel;
    std::vector<int> n_obs;
    int nperiod;
};

struct ResidualStats {
    real_t mean;
    real_t stddev;
};

class SrcRec {
public:
    // Unified accessor: indexed by WaveType (Rayleigh/Love) and SurfType (phase/group).
    // Four independent instances are created lazily on first access.
    static SrcRec& SR(WaveType wt, SurfType vt) {
        static SrcRec insts[2][2];
        return insts[static_cast<int>(wt)][static_cast<int>(vt)];
    }

    // Backward-compatible aliases (Rayleigh-only call sites)
    static SrcRec& SR_ph() { return SR(WaveType::RL, SurfType::PH); }
    static SrcRec& SR_gr() { return SR(WaveType::RL, SurfType::GR); }

    static Stations& stas() {
        static Stations inst;
        return inst;
    }

    // Build the shared station list from all loaded SrcRec instances.
    // If multiple tables are loaded, the station set is the intersection
    // across them; otherwise it is the union of the loaded ones.
    // Call after all active SR(wt, vt).load() calls.
    static void build_stas();

    SrcRec() = default;
    SrcRec(const SrcRec&)            = delete;
    SrcRec& operator=(const SrcRec&) = delete;
    ~SrcRec() { release_shm(); }

    // Load source-receiver table from CSV.
    // Rank 0 reads the file; each field gets its own per-node MPI shared-memory
    // window via Parallel::alloc_shared (1-D for numeric, 2-D for strings).
    // Must call release_shm() before MPI_Finalize().
    // The period column is always validated (finite, > 0); with check_obs the
    // observed tt and vel columns are too. Aborts all ranks on invalid rows.
    void load(const std::string& filepath, bool check_obs = false);

    // Log an error listing rows whose value in `col` is non-finite or <= 0.
    // Returns the number of such rows (0 = column is valid).
    int check_positive(const real_t* col, const std::string& col_name) const;

    // gather synthetic travel times
    void gather_syn_tt();

    // Compute mean and standard deviation of travel-time residuals
    // (syn_data - observed tt) across all events held locally, then
    // reduce across all MPI ranks to give global statistics.
    // Returns {mean, std}. Requires gather_syn_tt() to have been called.
    ResidualStats compute_residual_stats();

    // Free all shared-memory windows. Call before MPI_Finalize().
    void release_shm();

    // write the source-receiver table to a CSV file (for debugging)
    void write(const std::string& filepath, const bool is_fwd = false);

    inline int n_obs() const { return n_obs_; }
    inline int nsrc_total() const { return nsrc_total_; }

    // Numeric fields — 1-D shared pointers; access as stla[i], stlo[i], ...
    real_t* stla      = nullptr;
    real_t* stlo      = nullptr;
    real_t* evla      = nullptr;
    real_t* evlo      = nullptr;
    real_t* dist      = nullptr;
    real_t* period_all = nullptr;
    real_t* tt        = nullptr;
    real_t* vel       = nullptr;
    real_t* weight    = nullptr;
    Eigen::VectorX<real_t> tt_fwd;  // forward-modeled travel time (computed in surfdisp.cpp)
    Eigen::VectorX<real_t> vel_fwd;  // for convenient access in Eigen

    std::vector<std::string> evtname;  
    std::vector<std::string> staname;
    // std::map<std::string, event_info> events;  // map from event name to list of row indices for that event
    std::map<std::string, event_info> events_local; 
    std::vector<std::string> src_name_list_local;
    std::vector<std::string> src_name_list;
    PeriodInfo periods_info;

private:
    // One MPI_Win per numeric field
    MPI_Win win_stla_      = MPI_WIN_NULL;
    MPI_Win win_stlo_      = MPI_WIN_NULL;
    MPI_Win win_evla_      = MPI_WIN_NULL;
    MPI_Win win_evlo_      = MPI_WIN_NULL;
    MPI_Win win_dist_      = MPI_WIN_NULL;
    MPI_Win win_period_    = MPI_WIN_NULL;
    MPI_Win win_tt_        = MPI_WIN_NULL;
    MPI_Win win_vel_       = MPI_WIN_NULL;
    MPI_Win win_weight_    = MPI_WIN_NULL;

    // header name
    std::vector<std::string> header_names = {
        "tt", "staname", "stla", "stlo",
        "evtname", "evla", "evlo", "period",
        "weight", "dist", "vel"
    };

    void get_events();
    void get_periods();
    std::string filepath_;  // CSV path, for diagnostics
    int nsrc_total_ = 0;
    // Number of rows after load()
    int n_obs_ = 0;

};
