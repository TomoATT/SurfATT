#pragma once

#include "config.h"

#include <H5Cpp.h>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <string>
#include <vector>
#include <stdexcept>
#include <type_traits>

// ---------------------------------------------------------------------------
// Compile-time HDF5 type trait
// ---------------------------------------------------------------------------
template<typename T> inline H5::PredType h5_type_of();
template<> inline H5::PredType h5_type_of<int>()           { return H5::PredType::NATIVE_INT; }
template<> inline H5::PredType h5_type_of<float>()         { return H5::PredType::NATIVE_FLOAT; }
template<> inline H5::PredType h5_type_of<real_t>()        { return H5::PredType::NATIVE_DOUBLE; }
template<> inline H5::PredType h5_type_of<real2_t>()       { return H5::PredType::NATIVE_LDOUBLE; }

// ---------------------------------------------------------------------------
// H5IO — thin RAII wrapper around H5::H5File
//
// Usage (write):
//   H5IO f("model.h5", H5IO::TRUNC);
//   f.write_scalar("niter", 40);
//   f.write_vector("depth", v);
//   f.write_matrix("vel",   M);
//   f.write_volume("rho",   data, ni, nj, nk);
//   f.write_attr  ("/",     "creator", "SurfATT");
//
// Usage (read):
//   H5IO f("model.h5", H5IO::RDONLY);
//   int niter = f.read_scalar<int>("niter");
//   auto v    = f.read_vector<double>("depth");
//   auto M    = f.read_matrix<double>("vel");     // returns MatrixXd (col-major)
//   hsize_t ni,nj,nk;
//   auto d    = f.read_volume<double>("rho", ni, nj, nk);  // flat [ni*nj*nk]
// ---------------------------------------------------------------------------
class H5IO {
public:
    enum Mode { RDONLY, RDWR, TRUNC };

    explicit H5IO(const std::string &path, Mode mode = RDONLY);
    ~H5IO() = default;

    // ---- Scalar ------------------------------------------------------------
    template<typename T>
    void write_scalar(const std::string &name, T value) {
        ensure_not_readonly();
        remove_if_exists(name);
        hsize_t dims[1] = {1};
        H5::DataSpace sp(1, dims);
        H5::DataSet ds = file_.createDataSet(name, h5_type_of<T>(), sp);
        ds.write(&value, h5_type_of<T>());
    }

    template<typename T>
    T read_scalar(const std::string &name) const {
        H5::DataSet ds = file_.openDataSet(name);
        T value{};
        ds.read(&value, h5_type_of<T>());
        return value;
    }

    // ---- 1-D vector (std::vector<T> or Eigen vector) ----------------------
    template<typename T>
    void write_vector(const std::string &name, const std::vector<T> &v) {
        ensure_not_readonly();
        remove_if_exists(name);
        hsize_t dims[1] = {static_cast<hsize_t>(v.size())};
        H5::DataSpace sp(1, dims);
        H5::DataSet ds = file_.createDataSet(name, h5_type_of<T>(), sp);
        ds.write(v.data(), h5_type_of<T>());
    }

    template<typename Derived>
    void write_vector(const std::string &name,
                      const Eigen::DenseBase<Derived> &v) {
        using T = typename Derived::Scalar;
        ensure_not_readonly();
        remove_if_exists(name);
        // Force evaluation to a contiguous column-vector
        Eigen::Matrix<T, Eigen::Dynamic, 1> tmp = v;
        hsize_t dims[1] = {static_cast<hsize_t>(tmp.size())};
        H5::DataSpace sp(1, dims);
        H5::DataSet ds = file_.createDataSet(name, h5_type_of<T>(), sp);
        ds.write(tmp.data(), h5_type_of<T>());
    }

    template<typename T>
    std::vector<T> read_vector(const std::string &name) const {
        H5::DataSet   ds = file_.openDataSet(name);
        H5::DataSpace sp = ds.getSpace();
        if (sp.getSimpleExtentNdims() != 1)
            throw std::runtime_error(
                "H5IO::read_vector: dataset '" + name + "' is not 1-D");
        hsize_t n = 0;
        sp.getSimpleExtentDims(&n, nullptr);
        std::vector<T> v(n);
        ds.read(v.data(), h5_type_of<T>());
        return v;
    }

    // ---- 2-D matrix (Eigen, stored row-major on disk) ----------------------
    template<typename T>
    void write_matrix(const std::string &name,
                      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &M) {
        ensure_not_readonly();
        remove_if_exists(name);
        // Write as row-major so dimensions match intuition (rows × cols)
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Rm = M;
        hsize_t dims[2] = {static_cast<hsize_t>(M.rows()),
                            static_cast<hsize_t>(M.cols())};
        H5::DataSpace sp(2, dims);
        H5::DataSet   ds = file_.createDataSet(name, h5_type_of<T>(), sp);
        ds.write(Rm.data(), h5_type_of<T>());
    }

    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    read_matrix(const std::string &name) const {
        H5::DataSet   ds = file_.openDataSet(name);
        H5::DataSpace sp = ds.getSpace();
        hsize_t dims[2]  = {0, 0};
        sp.getSimpleExtentDims(dims, nullptr);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Rm(dims[0], dims[1]);
        ds.read(Rm.data(), h5_type_of<T>());
        return Rm;  // implicitly converts to ColMajor if assigned to MatrixX*
    }

    // ---- 3-D Eigen Tensor or TensorMap (RowMajor, written as ni × nj × nk volume) ------
    template<typename TensorLike>
    void write_tensor(const std::string &name, const TensorLike &t) {
        using T = typename TensorLike::Scalar;
        ensure_not_readonly();
        remove_if_exists(name);
        hsize_t dims[3] = {static_cast<hsize_t>(t.dimension(0)),
                           static_cast<hsize_t>(t.dimension(1)),
                           static_cast<hsize_t>(t.dimension(2))};
        H5::DataSpace sp(3, dims);
        H5::DataSet   ds = file_.createDataSet(name, h5_type_of<T>(), sp);
        ds.write(t.data(), h5_type_of<T>());
    }

    template<typename T>
    Eigen::Tensor<T, 3, Eigen::RowMajor> read_tensor(const std::string &name) const {
        hsize_t ni = 0, nj = 0, nk = 0;
        auto data = read_volume<T>(name, ni, nj, nk);
        Eigen::Tensor<T, 3, Eigen::RowMajor> t(static_cast<int>(ni),
                                                static_cast<int>(nj),
                                                static_cast<int>(nk));
        std::copy(data.begin(), data.end(), t.data());
        return t;
    }

    // ---- 3-D volume (flat array, dims: ni × nj × nk, row-major on disk) ---
    // data must have exactly ni*nj*nk elements.

    template<typename T>
    void write_volume(const std::string &name,
                      const T *data, hsize_t ni, hsize_t nj, hsize_t nk) {
        ensure_not_readonly();
        remove_if_exists(name);
        hsize_t dims[3] = {ni, nj, nk};
        H5::DataSpace sp(3, dims);
        H5::DataSet   ds = file_.createDataSet(name, h5_type_of<T>(), sp);
        ds.write(data, h5_type_of<T>());           
    }

    // Overload accepting std::vector<T> — delegates to the raw-pointer overload.
    template<typename T>
    void write_volume(const std::string &name,
                      const std::vector<T> &data,
                      hsize_t ni, hsize_t nj, hsize_t nk) {
        if (data.size() != ni * nj * nk)
            throw std::invalid_argument(
                "H5IO::write_volume: data.size() != ni*nj*nk");
        write_volume(name, data.data(), ni, nj, nk);
    }

    // Read a 3-D dataset; ni/nj/nk are set to the actual dataset dimensions.
    template<typename T>
    std::vector<T> read_volume(const std::string &name,
                               hsize_t &ni, hsize_t &nj, hsize_t &nk) const {
        H5::DataSet   ds = file_.openDataSet(name);
        H5::DataSpace sp = ds.getSpace();
        if (sp.getSimpleExtentNdims() != 3)
            throw std::runtime_error(
                "H5IO::read_volume: dataset '" + name + "' is not 3-D");
        hsize_t dims[3] = {0, 0, 0};
        sp.getSimpleExtentDims(dims, nullptr);
        ni = dims[0]; nj = dims[1]; nk = dims[2];
        std::vector<T> data(ni * nj * nk);
        ds.read(data.data(), h5_type_of<T>());
        return data;
    }

    // Convenience: index helper for flat 3-D arrays (row-major)
    // Usage: data[idx3(ii, jj, kk, nj, nk)]
    static constexpr hsize_t idx3(hsize_t ii, hsize_t jj, hsize_t kk,
                                   hsize_t nj, hsize_t nk) noexcept {
        return ii * nj * nk + jj * nk + kk;
    }

    // ---- String attribute on a group/dataset -------------------------------
    void write_attr(const std::string &obj_path,
                    const std::string &attr_name,
                    const std::string &value);

    std::string read_attr(const std::string &obj_path,
                          const std::string &attr_name) const;

    // ---- Existence check ---------------------------------------------------
    bool exists(const std::string &name) const;

    // ---- Underlying file handle (for advanced use) -------------------------
    H5::H5File &file() { return file_; }
    const H5::H5File &file() const { return file_; }

private:
    H5::H5File file_;
    bool       readonly_{false};

    void remove_if_exists(const std::string& name) {
        if (file_.nameExists(name)) {
            file_.unlink(name);
        }
    }

    void ensure_not_readonly() const {
        if (readonly_)
            throw std::logic_error("H5IO: file opened read-only");
    }
};
