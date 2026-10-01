#include "input_params.h"
#include "parallel.h"
#include "src_rec.h"
#include "logger.h"
#include "config.h"
#include "utils.h"
#include "model_grid.h"
#include "argparser.h"
#include "topo.h"
#include "surf_grid.h"
#include "preproc.h"
#include "decomposer.h"
#include "inversion.h"

#include <algorithm>


int main(int argc, char* argv[])
{
    // parse options    
    argparse_tomo(argc, argv);

    // initialise MPI
    Parallel::init();

    // read input parameters
    InputParams::read(input_file);
    auto& IP = InputParams::IP();

    // initialize logger
    ATTLogger::init(
        fmt::format("{}/{}", IP.output().output_path, LOG_FNAME),
        IP.output().log_level,
        false
    );

    // load source-receiver tables into shared memory; tt/vel are observations
    // (and so validated) only when inverting
    for (auto [wt, vt] : IP.data().active_data)
        SrcRec::SR(wt, vt).load(IP.data().file_of(wt, vt), run_mode == INVERSION_MODE);
    SrcRec::build_stas();

    // build model grid
    ModelGrid::init();
    auto &mg = ModelGrid::MG();

    // build initial model
    mg.build_init_model();

    // initialize decomposer (for parallel execution)
    Decomposer::DCP();

    // load topography if needed
    if (IP.topo().is_consider_topo) {
        // load topography
        Topography::read(IP.topo().topo_file);
    }
    // build surface grid and compute reference travel times
    for (auto [wt, vt] : IP.data().active_data)
        SurfGrid::SG(wt, vt).build_media();

    Inversion::init();

    // run forward and adjoint calculations
    if (run_mode == FORWARD_ONLY) {
        Inversion::INV().run_forward();
    } else {
        Inversion::INV().run_inversion();
    }

    for (auto [wt, vt] : IP.data().active_data)
        SrcRec::SR(wt, vt).release_shm();

    double total_time = Parallel::mpi().elapsed_time();
    ATTLogger::logger().Info(fmt::format("Total runtime: {:.2f} seconds", total_time), MODULE_MAIN);

    return 0;
}