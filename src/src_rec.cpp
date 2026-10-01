#include "src_rec.h"
#include "parallel.h"
#include "utils.h"
#include "config.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>


// ---------------------------------------------------------------------------
// Load the source-receiver observation table from CSV and place numeric fields
// into MPI shared-memory windows.
// Design choice:
//   - Only rank 0 performs file I/O (single reader, deterministic parsing).
//   - Other ranks obtain the same data via broadcast + shared-memory sync,
//     avoiding duplicated memory footprint per rank.
void SrcRec::load(const std::string& filepath, bool check_obs)
{
    auto& mpi = Parallel::mpi();
    // auto& IP = InputParams::IP();
    auto& logger = ATTLogger::logger();
    filepath_ = filepath;

    std::vector<real_t> v_stla, v_stlo, v_evla, v_evlo, v_dist,
                        v_period, v_tt, v_vel, v_weight;
    std::vector<std::string> v_evtname, v_staname;

    logger.Info("Loading source-receiver table from " + filepath, MODULE_SRCREC);

    // 1. Rank 0 reads the CSV
    if (mpi.is_main()) {
        rapidcsv::Document doc;
        try {
            doc = rapidcsv::Document(filepath, rapidcsv::LabelParams(0, -1));
        } catch (const std::ios_base::failure &e) {
            logger.Error(fmt::format(
                "SrcRec::load: failed to open file {}: {}", filepath, e.what()
            ), MODULE_SRCREC);
            mpi.abort(EXIT_FAILURE);
        }
        n_obs_ = static_cast<int>(doc.GetRowCount());

        v_stla    = doc.GetColumn<real_t>("stla");
        v_stlo    = doc.GetColumn<real_t>("stlo");
        v_evla    = doc.GetColumn<real_t>("evla");
        v_evlo    = doc.GetColumn<real_t>("evlo");
        v_period  = doc.GetColumn<real_t>("period");
        v_tt      = doc.GetColumn<real_t>("tt");
        evtname = doc.GetColumn<std::string>("evtname");
        staname = doc.GetColumn<std::string>("staname");

        // Optional "weight" column; if missing, fill with 1.0
        try{
            v_weight  = doc.GetColumn<real_t>("weight");
        } catch (const std::exception& e) {
            // If "weight" column is missing, fill with 1.0
            v_weight.resize(n_obs_, _1_CR);
        }

        // Optional "dist" column; if missing, compute from stla/stlo/evla/evlo
        try {
            v_dist    = doc.GetColumn<real_t>("dist");
        } catch (const std::exception& e) {
            v_dist.resize(n_obs_);
            for (int i = 0; i < n_obs_; ++i) {
                v_dist[i] = gps2dist(v_stla[i], v_stlo[i], v_evla[i], v_evlo[i]);
            }
        }

        // Optional "vel" column; if missing, compute from dist and tt
        try {
            v_vel     = doc.GetColumn<real_t>("vel");
        } catch (const std::exception& e) {
            v_vel.resize(n_obs_);
            for (int i = 0; i < n_obs_; ++i) {
                v_vel[i] = (v_tt[i] > 0) ? (v_dist[i] / v_tt[i]) : 0.0;
            }
        }
    }
    mpi.barrier();

    // 2. Broadcast row count
    mpi.bcast(n_obs_);

    // 3) Allocate one shared-memory window per numeric field.
    //    This allows all local ranks on the node to access a single copy
    //    instead of storing duplicated arrays.
    mpi.alloc_shared(n_obs_, stla, win_stla_);
    mpi.alloc_shared(n_obs_, stlo, win_stlo_);
    mpi.alloc_shared(n_obs_, evla, win_evla_);
    mpi.alloc_shared(n_obs_, evlo, win_evlo_);
    mpi.alloc_shared(n_obs_, dist, win_dist_);
    mpi.alloc_shared(n_obs_, period_all, win_period_);
    mpi.alloc_shared(n_obs_, tt, win_tt_);
    mpi.alloc_shared(n_obs_, vel, win_vel_);
    mpi.alloc_shared(n_obs_, weight, win_weight_);
    if (mpi.is_main()) {
        std::copy(v_stla.begin(), v_stla.end(), stla);
        std::copy(v_stlo.begin(), v_stlo.end(), stlo);
        std::copy(v_evla.begin(), v_evla.end(), evla);
        std::copy(v_evlo.begin(), v_evlo.end(), evlo);
        std::copy(v_dist.begin(), v_dist.end(), dist);
        std::copy(v_period.begin(), v_period.end(), period_all);
        std::copy(v_tt.begin(), v_tt.end(), tt);
        std::copy(v_vel.begin(), v_vel.end(), vel);
        std::copy(v_weight.begin(), v_weight.end(), weight);
    }
    mpi.barrier();

    // After rank 0 populates the shared buffers, synchronize visibility so
    // every rank sees the same finalized content before downstream processing.
    mpi.sync_from_main_rank(stla, n_obs_);
    mpi.sync_from_main_rank(stlo, n_obs_);
    mpi.sync_from_main_rank(evla, n_obs_);
    mpi.sync_from_main_rank(evlo, n_obs_);
    mpi.sync_from_main_rank(dist, n_obs_);
    mpi.sync_from_main_rank(period_all, n_obs_);
    mpi.sync_from_main_rank(tt, n_obs_);
    mpi.sync_from_main_rank(vel, n_obs_);
    mpi.sync_from_main_rank(weight, n_obs_);

    // Validate before get_periods(): its std::sort needs NaN-free periods.
    // tt/vel are only checked when they are observations; forward runs may
    // carry placeholders there. Every rank sees the same shared data, so all
    // reach the same verdict; the barrier lets rank 0 flush its log first.
    int n_bad_cols = (check_positive(period_all, "period") > 0);
    if (check_obs) {
        n_bad_cols += (check_positive(tt, "tt") > 0);
        n_bad_cols += (check_positive(vel, "vel") > 0);
    }
    if (n_bad_cols > 0) {
        logger.Error(fmt::format("SrcRec::load: {} column(s) of {} contain invalid values; see messages above.",
            n_bad_cols, filepath), MODULE_SRCREC);
        mpi.barrier();
        mpi.abort(EXIT_FAILURE);
    }

    // Build derived metadata used by inversion/forward steps:
    //   - event-wise receiver index lists
    //   - per-period statistics (counts and mean velocities)
    get_periods();
    get_events();
}

// Rows are reported as CSV line numbers: data row index + 2 (header is line 1).
int SrcRec::check_positive(const real_t* col, const std::string& col_name) const {
    constexpr int MAX_BAD_ROWS_SHOWN = 10;
    std::vector<int> rows;
    for (int i = 0; i < n_obs_; ++i) {
        if (!std::isfinite(col[i]) || col[i] <= _0_CR) rows.push_back(i);
    }
    if (rows.empty()) return 0;

    std::string msg = fmt::format("{}: '{}' must be finite and > 0; {} row(s):",
        filepath_, col_name, rows.size());
    for (size_t i = 0; i < rows.size() && i < MAX_BAD_ROWS_SHOWN; ++i) {
        msg += fmt::format(" line {} ({})", rows[i] + 2, col[rows[i]]);
    }
    if (rows.size() > MAX_BAD_ROWS_SHOWN) msg += " ...";
    ATTLogger::logger().Error(msg, MODULE_SRCREC);
    return static_cast<int>(rows.size());
}

// Build an event -> observation-index mapping from evtname and distribute
// event payloads to destination ranks.
//
// Distribution strategy:
//   - rank 0 constructs the global event map.
//   - each event is assigned to a rank via select_rank_for_src(i_src).
//   - assigned rank stores the event in events_local / src_name_list_local.
void SrcRec::get_events(){
    auto& mpi = Parallel::mpi();
    auto& logger = ATTLogger::logger();

    std::vector<std::string> valid_event_keys;
    std::map<std::string, event_info> events;
    if (mpi.is_main()){
        valid_event_keys.resize(evtname.size());
        std::copy(evtname.begin(), evtname.end(), valid_event_keys.begin());
        // Extract sorted unique event names from the per-observation evtname list.
        std::sort(valid_event_keys.begin(), valid_event_keys.end());
        valid_event_keys.erase(std::unique(valid_event_keys.begin(), valid_event_keys.end()), valid_event_keys.end());        

        for (const auto& src : valid_event_keys) {
            for (int iper = 0; iper < periods_info.nperiod; ++iper) {
                event_info info;
                for (int i = 0; i < n_obs_; ++i) {
                    
                    if (evtname[i] == src && real_t_equal(periods_info.periods(iper), period_all[i])) {
                        info.rec_indices.push_back(i);
                        info.evla = evla[i];
                        info.evlo = evlo[i];
                        info.period = period_all[i];
                        info.iper = iper;
                    }
                }
                int n_rec = info.rec_indices.size();
                if (n_rec > 0) {
                    info.syn_data.setZero(n_rec);  // placeholder for synthetic data
                    std::string key = fmt::format("{}_{:d}", src, iper);
                    events[key] = info;
                    src_name_list.push_back(key);
                }
            }
        }
        nsrc_total_ = static_cast<int>(src_name_list.size());
    }
    mpi.barrier();
    mpi.bcast(nsrc_total_);
    if (!mpi.is_main()) src_name_list.resize(nsrc_total_);
    for (int i = 0; i < nsrc_total_; ++i) {
        mpi.bcast(src_name_list[i]);
    }

    // Distribute event records to their target ranks.
    // We broadcast event names (small metadata), then send larger payloads
    // (evla/evlo/receiver-index list) only to the selected destination rank.
    logger.Info(fmt::format(
        "Distributing {} events to {} ranks", nsrc_total_, mpi.size()),
        MODULE_SRCREC
    );
    for (int i_src = 0; i_src < nsrc_total_; i_src++) {
        int dst_rank = mpi.select_rank_for_src(i_src);
        
        std::string src_name;
        if (mpi.is_main()) src_name = src_name_list[i_src];
        mpi.bcast(src_name);

        if (mpi.is_main()) {
            if (dst_rank == 0) {
                src_name_list_local.push_back(src_name);
                events_local[src_name] = events[src_name];
            } else {
                event_info info = events[src_name];
                int n_rec = static_cast<int>(info.rec_indices.size());
                mpi.send(&info.evla, 1, dst_rank);
                mpi.send(&info.evlo, 1, dst_rank);
                mpi.send(&info.period, 1, dst_rank);
                mpi.send(&info.iper, 1, dst_rank);
                mpi.send(&n_rec, 1, dst_rank);
                mpi.send(info.rec_indices.data(), n_rec, dst_rank);
                mpi.send(info.syn_data.data(), n_rec, dst_rank);  
            }
        } else if (mpi.rank() == dst_rank) {
            src_name_list_local.push_back(src_name);
            event_info info;
            int n_rec = 0;
            mpi.recv(&info.evla, 1, 0);
            mpi.recv(&info.evlo, 1, 0);
            mpi.recv(&info.period, 1, 0);
            mpi.recv(&info.iper, 1, 0);
            mpi.recv(&n_rec, 1, 0);
            info.rec_indices.resize(n_rec);
            info.syn_data.resize(n_rec);
            mpi.recv(info.rec_indices.data(), n_rec, 0);
            mpi.recv(info.syn_data.data(), n_rec, 0);
            events_local[src_name] = info;
        }
    }
    mpi.barrier();
}

// Compute unique period values and average velocity for each period.
// Current implementation performs this aggregation on rank 0 only.
void SrcRec::get_periods(){
    auto& mpi = Parallel::mpi();

    if (mpi.is_main()){
        std::vector<real_t> periods_vec(n_obs_);
        std::copy(period_all, period_all + n_obs_, periods_vec.begin());
        std::sort(periods_vec.begin(), periods_vec.end());
        periods_vec.erase(std::unique(periods_vec.begin(), periods_vec.end()), periods_vec.end());
        periods_info.nperiod = static_cast<int>(periods_vec.size());
        periods_info.periods = Eigen::Map<Eigen::VectorX<real_t>>(periods_vec.data(), periods_info.nperiod);
        periods_info.meanvel = Eigen::VectorX<real_t>::Zero(periods_info.nperiod);

        for (int i = 0; i < periods_info.nperiod; ++i) {
            real_t per = periods_info.periods(i);
            int count = 0;
            real_t sum_vel = _0_CR;
            for (int j = 0; j < n_obs_; ++j) {
                if (std::abs(period_all[j] - per) < 1e-6) {
                    count++;
                    sum_vel += vel[j];
                }
            }
            periods_info.n_obs.push_back(count);
            periods_info.meanvel(i) = sum_vel / count;
        }
    }
    mpi.barrier();
    mpi.bcast(periods_info.nperiod);
    if (!mpi.is_main()) {
        periods_info.periods.resize(periods_info.nperiod);
        periods_info.meanvel.resize(periods_info.nperiod);
    }
    mpi.bcast(periods_info.periods.data(), periods_info.nperiod);
    mpi.bcast(periods_info.meanvel.data(), periods_info.nperiod);
}

// ---------------------------------------------------------------------------
// Build the global station list used by the workflow.
// Rules:
//   - If both phase and group datasets are available, keep the intersection
//     of station names to ensure consistency across data types.
//   - If only one dataset is available, use all stations from that dataset.
void SrcRec::build_stas()
{
    auto& mpi = Parallel::mpi();
    auto& st  = stas();
    auto& IP  = InputParams::IP();

    if (mpi.is_main()) {
        auto make_map = [](const SrcRec& sr) {
            std::map<std::string, std::pair<real_t, real_t>> m;
            for (int i = 0; i < sr.n_obs(); ++i)
                m.try_emplace(sr.staname[i], sr.stla[i], sr.stlo[i]);
            return m;
        };

        // Collect per-table maps for every (wt, vt) actually loaded
        std::vector<std::map<std::string, std::pair<real_t, real_t>>> maps;
        for (auto [wt, vt] : IP.data().active_data) {
            auto& sr = SR(wt, vt);
            if (sr.n_obs() > 0) maps.push_back(make_map(sr));
        }

        std::map<std::string, std::pair<real_t, real_t>> merged;
        if (maps.size() == 1) {
            merged = std::move(maps.front());
        } else if (maps.size() > 1) {
            // Intersection across all loaded tables
            merged = maps.front();
            for (size_t i = 1; i < maps.size(); ++i) {
                std::map<std::string, std::pair<real_t, real_t>> next;
                for (auto& [name, coords] : merged)
                    if (maps[i].count(name)) next[name] = coords;
                merged = std::move(next);
            }
        }

        st.stnm.clear(); st.stla.clear(); st.stlo.clear();
        for (auto& [name, coords] : merged) {
            st.stnm.push_back(name);
            st.stla.push_back(coords.first);
            st.stlo.push_back(coords.second);
        }
    }
    mpi.barrier();

    // Broadcast to all ranks
    int nsta = mpi.is_main() ? static_cast<int>(st.stnm.size()) : 0;
    mpi.bcast(nsta);
    if (!mpi.is_main()) {
        st.stnm.resize(nsta);
        st.stla.resize(nsta);
        st.stlo.resize(nsta);
    }
    for (int i = 0; i < nsta; ++i)
        mpi.bcast(st.stnm[i]);
    mpi.bcast(st.stla.data(), nsta);
    mpi.bcast(st.stlo.data(), nsta);
}

void SrcRec::gather_syn_tt()
{
    auto &mpi = Parallel::mpi();
    if (mpi.is_main()) tt_fwd.setZero(n_obs_);

    for (int i_src = 0; i_src < nsrc_total_; i_src++) {
        int dst_rank = mpi.select_rank_for_src(i_src);
        std::string src_name;
        if (mpi.is_main()) src_name = src_name_list[i_src];
        mpi.bcast(src_name);

        if (mpi.is_main()) {
            int n_rec;
            if (dst_rank == 0) {
                auto& info = events_local[src_name];
                n_rec = static_cast<int>(info.rec_indices.size());
                for (int i = 0; i < n_rec; ++i) {
                    int rec_idx = info.rec_indices[i];
                    tt_fwd(rec_idx) = info.syn_data[i];
                }
            } else {
                mpi.recv(&n_rec, 1, dst_rank);
                Eigen::VectorX<real_t> syn_tt(n_rec);
                std::vector<int> rec_indices(n_rec);
                mpi.recv(syn_tt.data(), n_rec, dst_rank);
                mpi.recv(rec_indices.data(), n_rec, dst_rank);
                for (int i = 0; i < n_rec; ++i) {
                    int rec_idx = rec_indices[i];
                    tt_fwd(rec_idx) = syn_tt(i);
                }
            }
        } else if (mpi.rank() == dst_rank) {
            auto& info = events_local[src_name];
            int n_rec = static_cast<int>(info.rec_indices.size());
            mpi.send(&n_rec, 1, 0);
            mpi.send(info.syn_data.data(), n_rec, 0);
            mpi.send(info.rec_indices.data(), n_rec, 0);
        }
    }
    mpi.barrier();

    if (mpi.is_main()) {
        Eigen::Map<Eigen::VectorX<real_t>> dist_map(dist, n_obs_);
        vel_fwd = dist_map.array() / tt_fwd.array();
    }
}


ResidualStats SrcRec::compute_residual_stats() {
    auto &mpi = Parallel::mpi();

    real_t local_n   = _0_CR;
    real_t local_sum = _0_CR;
    real_t local_sq  = _0_CR;

    for (const auto& [name, info] : events_local) {
        const int n_rec = static_cast<int>(info.rec_indices.size());
        for (int i = 0; i < n_rec; ++i) {
            const real_t residual = info.syn_data[i] - tt[info.rec_indices[i]];
            local_n   += _1_CR;
            local_sum += residual;
            local_sq  += residual * residual;
        }
    }

    mpi.sum_all_all_inplace(local_n);
    mpi.sum_all_all_inplace(local_sum);
    mpi.sum_all_all_inplace(local_sq);

    if (local_n == _0_CR)
        return {_0_CR, _0_CR};

    const real_t mean = local_sum / local_n;
    const real_t var  = local_sq  / local_n - mean * mean;
    const real_t stddev = std::sqrt(std::max(var, _0_CR));
    return {mean, stddev};  // ResidualStats aggregate init
}

// Write the current observation table to CSV.
// If is_fwd == true, output forward-modeled fields (tt_fwd / vel_fwd);
// otherwise output observed/reference fields (tt / vel).
void SrcRec::write(const std::string& filepath, const bool is_fwd){
    if (n_obs_ == 0) {
        throw std::runtime_error("SrcRec::write: no observations to write");
    }
    auto &mpi = Parallel::mpi();
    auto &logger = ATTLogger::logger();

    if (mpi.is_main()) {
        rapidcsv::Document doc;
        if (is_fwd){
            if (tt_fwd.size() == 0) {
                logger.Error("Forward-modeled travel times (tt_fwd) are not assigned", MODULE_SRCREC);
                mpi.abort(EXIT_FAILURE);
            }
            doc.SetColumn<std::string> (0,  fmt_col(tt_fwd.data(), n_obs_, 6));
        } else {
            doc.SetColumn<std::string> (0,  fmt_col(tt,     n_obs_, 6));
        }
        doc.SetColumn<std::string> (1,  staname);
        doc.SetColumn<std::string> (2,  fmt_col(stla,       n_obs_, 4));
        doc.SetColumn<std::string> (3,  fmt_col(stlo,       n_obs_, 4));
        doc.SetColumn<std::string> (4,  evtname);
        doc.SetColumn<std::string> (5,  fmt_col(evla,       n_obs_, 4));
        doc.SetColumn<std::string> (6,  fmt_col(evlo,       n_obs_, 4));
        doc.SetColumn<std::string> (7,  fmt_col(period_all, n_obs_, 2));
        doc.SetColumn<std::string> (8,  fmt_col(weight,     n_obs_, 4));
        doc.SetColumn<std::string> (9,  fmt_col(dist,       n_obs_, 3));
        if (is_fwd) {
            if (vel_fwd.size() == 0) {
                logger.Error("Forward-modeled velocities (vel_fwd) are not assigned", MODULE_SRCREC);
                mpi.abort(EXIT_FAILURE);
            }
            doc.SetColumn<std::string> (10, fmt_col(vel_fwd.data(), n_obs_, 6));
        } else {
            doc.SetColumn<std::string> (10, fmt_col(vel,        n_obs_, 6));
        }
        doc.SetColumnName(0,  "tt");
        doc.SetColumnName(1,  "staname");
        doc.SetColumnName(2,  "stla");
        doc.SetColumnName(3,  "stlo");
        doc.SetColumnName(4,  "evtname");
        doc.SetColumnName(5,  "evla");
        doc.SetColumnName(6,  "evlo");
        doc.SetColumnName(7,  "period");
        doc.SetColumnName(8,  "weight");
        doc.SetColumnName(9,  "dist");
        doc.SetColumnName(10, "vel");
        doc.Save(filepath);
    }
    mpi.barrier();
}

// ---------------------------------------------------------------------------
// Release all MPI shared-memory windows allocated in load().
// Must be called when SrcRec data are no longer needed to avoid SHM leaks.
void SrcRec::release_shm()
{
    auto& mpi = Parallel::mpi();
    mpi.free_shared(stla,       win_stla_);
    mpi.free_shared(stlo,       win_stlo_);
    mpi.free_shared(evla,       win_evla_);
    mpi.free_shared(evlo,       win_evlo_);
    mpi.free_shared(dist,       win_dist_);
    mpi.free_shared(period_all, win_period_);
    mpi.free_shared(tt,         win_tt_);
    mpi.free_shared(vel,        win_vel_);
    mpi.free_shared(weight,     win_weight_);
}
