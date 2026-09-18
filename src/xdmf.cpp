#include "xdmf.h"
#include "config.h"
#include "input_params.h"
#include "parallel.h"

#include <spdlog/fmt/fmt.h>
#include <fstream>

namespace xdmf {

// Data layout in model_iter.h5: C row-major [ix][iy][iz], i.e. lon(ix) is the
// slowest-varying index and depth(iz) is the fastest.
//
// XDMF 3DRectMesh convention: Dimensions="A B C" means A is slowest (VTK Z),
// C is fastest (VTK X).  VXVYVZ lists the coordinate vector for the X
// direction first, then Y, then Z.
//
// To match our storage we set:
//   Dimensions = "nx ny nz"  (nx=lon slowest → VTK-Z, nz=depth fastest → VTK-X)
//   VXVYVZ[0] = z/depth  (nz elements, VTK-X)
//   VXVYVZ[1] = y/lat    (ny elements, VTK-Y)
//   VXVYVZ[2] = x/lon    (nx elements, VTK-Z)
//
// ParaView axis labels: X=depth, Y=lat, Z=lon.  Apply a Transform filter to
// rearrange axes for geographic display if needed.

void write_model_iter(const std::string &xdmf_path, int iter,
                      int last_grad_iter) {
    if (!Parallel::mpi().is_main()) return;

    auto &IP  = InputParams::IP();
    const std::string h5ref = MODEL_ITER_FNAME; // same directory as xdmf_path
    const int nx = ngrid_i, ny = ngrid_j, nz = ngrid_k;

    // Topology: "nx ny nz" → slowest(lon)→VTK-Z, fastest(depth)→VTK-X
    const std::string topo_dims = fmt::format("{} {} {}", nx, ny, nz);
    // Attribute DataItem matches topology memory layout
    const std::string attr_dims = topo_dims;
    const std::string di_hdr = fmt::format(
        "Dimensions=\"{}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\"",
        attr_dims);

    auto attr = [&](const std::string &name, const std::string &ds) {
        return fmt::format(
            "        <Attribute Name=\"{}\" AttributeType=\"Scalar\" Center=\"Node\">\n"
            "          <DataItem {}>{}:/{}</DataItem>\n"
            "        </Attribute>\n",
            name, di_hdr, h5ref, ds);
    };

    std::ofstream out(xdmf_path);
    out << "<?xml version=\"1.0\" ?>\n"
           "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
           "<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"2.0\">\n"
           "  <Domain>\n"
           "    <Grid Name=\"ModelIterations\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";

    for (int it = 0; it <= iter; ++it) {
        const std::string sfx = fmt::format("_{:03d}", it);
        out << fmt::format("      <Grid Name=\"iter_{:03d}\" GridType=\"Uniform\">\n", it)
            << fmt::format("        <Time Value=\"{}\"/>\n", it);

        if (it == 0) {
            // Topology and geometry defined once; subsequent grids reference them.
            out << fmt::format("        <Topology Name=\"topo\" TopologyType=\"3DRectMesh\" Dimensions=\"{}\"/>\n", topo_dims)
                << "        <Geometry Name=\"geom\" GeometryType=\"VXVYVZ\">\n"
                << fmt::format("          <DataItem Dimensions=\"{}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">{}:/z</DataItem>\n", nz, h5ref)
                << fmt::format("          <DataItem Dimensions=\"{}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">{}:/y_km</DataItem>\n", ny, h5ref)
                << fmt::format("          <DataItem Dimensions=\"{}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">{}:/x_km</DataItem>\n", nx, h5ref)
                << "        </Geometry>\n";
        } else {
            out << "        <xi:include xpointer=\"xpointer(//Grid[@Name='iter_000']/Topology)\"/>\n"
                   "        <xi:include xpointer=\"xpointer(//Grid[@Name='iter_000']/Geometry)\"/>\n";
        }

        // --- model fields ---
        out << attr("vs", "model_vs" + sfx);
        if (IP.inversion().use_alpha_beta_rho) {
            out << attr("vp",  "model_vp"  + sfx)
                << attr("rho", "model_rho" + sfx);
        }
        if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
            out << attr("gc", "model_gc" + sfx)
                << attr("gs", "model_gs" + sfx);
        }
        if (it <= last_grad_iter) {
            out << attr("grad_vs", "grad_vs" + sfx);
            if (IP.inversion().use_alpha_beta_rho) {
                out << attr("grad_vp",  "grad_vp"  + sfx)
                    << attr("grad_rho", "grad_rho" + sfx);
            }
            if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
                out << attr("grad_gc", "grad_gc" + sfx)
                    << attr("grad_gs", "grad_gs" + sfx);
            }
            if (IP.postproc().is_kden) {
                out << attr("kernel_density", "kernel_density" + sfx);
            }
        }
        out << "      </Grid>\n";
    }

    out << "    </Grid>\n"
           "  </Domain>\n"
           "</Xdmf>\n";
}

} // namespace xdmf
