#pragma once

#include "config.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <filesystem>
#include <iomanip>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

inline int I2V_common(int A, int B, int C, int ny, int nz) {
    return (A*nz*ny + B*nz + C);  // 3D vector to 1D array index
}

inline int I2V_common_2d(int A, int B, int ny) {
    return (A*ny + B);  // 2D vector to 1D array index
}

inline bool real_t_equal(real_t a, real_t b, real_t tol = 1e-6) {
    return std::abs(a - b) < tol;
}

// Bilinear interpolation at fractional indices (idx0+r1, idy0+r2)
inline real_t bilinear(const Eigen::MatrixXd& M,
                       int idx0, int idy0,
                       real_t r1, real_t r2)
{
    return (1-r1)*(1-r2)*M(idx0,  idy0  )
         + (1-r1)*  r2  *M(idx0,  idy0+1)
         +    r1 *(1-r2)*M(idx0+1,idy0  )
         +    r1 *  r2  *M(idx0+1,idy0+1);
}

template <typename T>
inline int locate_bissection(const T* valx, int n, T x){
    int low = 0;
    int high = n - 1;

    if (x < valx[0] || x > valx[n-1]) {
        return -1;
    }
    while (high - low > 1) {
        int mid = (high + low) / 2;
        if (valx[mid] > x)
            high = mid;
        else
            low = mid;
    }
    if (x == valx[0]) {
        return 0;
    } else if (x == valx[n-1]) {
        return n-2;
    } else {
        return low;
    }
}

inline real_t bilinear_interpolation(real_t* xarr, real_t* yarr, 
                                     int nx, int ny, real_t* values, 
                                     real_t x, real_t y){
    // convert values to 2D array
    std::vector<std::vector<real_t>> values_2d(nx, std::vector<real_t>(ny, 0.0));
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            values_2d[i][j] = values[I2V_common_2d(i, j, ny)];
        }
    }

    // find the index of x and y in xarr and yarr
    int x_idx = -1;
    int y_idx = -1;
    
    // find closest x
    for (int i = 0; i < nx-1; i++){
        if (x == xarr[i]){
            x_idx = i;
            break;
        } else if (x > xarr[i] && x < xarr[i+1]){
            x_idx = i;
            break;
        }
    }
    if (x == xarr[nx-1]){
        x_idx = nx-1;
    }

    // find closest y
    for (int i = 0; i < ny-1; i++){
        if (y == yarr[i]){
            y_idx = i;
            break;
        } else if (y > yarr[i] && y < yarr[i+1]){
            y_idx = i;
            break;
        }
    }
    if (y == yarr[ny-1]){
        y_idx = ny-1;
    }

    // if x or y is not in xarr or yarr, return -1
    if (x_idx == -1 || y_idx == -1){
        return NAN;
    }

    // if x and y are in xgrids and ygrids, return the value
    if (x_idx == nx-1 && y_idx == ny-1){
        return values_2d[x_idx][y_idx];
    }

    // if x is in xgrids and y is in ygrids, return the value
    if (x_idx == nx-1){
        return values_2d[x_idx][y_idx] + (y - yarr[y_idx]) * (values_2d[x_idx][y_idx+1] - values_2d[x_idx][y_idx]) / (yarr[y_idx+1] - yarr[y_idx]);
    }

    if (y_idx == ny-1){
        return values_2d[x_idx][y_idx] + (x - xarr[x_idx]) * (values_2d[x_idx+1][y_idx] - values_2d[x_idx][y_idx]) / (xarr[x_idx+1] - xarr[x_idx]);
    }

    // if x and y are not in xgrids and ygrids, return the interpolated value
    real_t x1 = xarr[x_idx];
    real_t x2 = xarr[x_idx+1];
    real_t y1 = yarr[y_idx];
    real_t y2 = yarr[y_idx+1];

    real_t f11 = values_2d[x_idx][y_idx];
    real_t f21 = values_2d[x_idx+1][y_idx];
    real_t f12 = values_2d[x_idx][y_idx+1];
    real_t f22 = values_2d[x_idx+1][y_idx+1];
    
    real_t result = f11 * (x2 - x) * (y2 - y) +
                        f21 * (x - x1) * (y2 - y) + 
                        f12 * (x2 - x) * (y - y1) + 
                        f22 * (x - x1) * (y - y1);
    result /= ((x2 - x1) * (y2 - y1));

    return result;
}

// Trilinear interpolation on a rectilinear grid.
// values is a flat array with layout (nx, ny, nz):
//   values[i*ny*nz + j*nz + k] = f(xarr[i], yarr[j], zarr[k])
// Returns NaN when (x,y,z) is out of bounds.
inline real_t trilinear_interpolation(
    const real_t* xarr, const real_t* yarr, const real_t* zarr,
    int nx, int ny, int nz,
    const real_t* values,
    real_t x, real_t y, real_t z)
{
    const int ix = locate_bissection(xarr, nx, x);
    const int iy = locate_bissection(yarr, ny, y);
    const int iz = locate_bissection(zarr, nz, z);
    if (ix < 0 || iy < 0 || iz < 0) return NAN;

    const real_t wx = (x - xarr[ix]) / (xarr[ix + 1] - xarr[ix]);
    const real_t wy = (y - yarr[iy]) / (yarr[iy + 1] - yarr[iy]);
    const real_t wz = (z - zarr[iz]) / (zarr[iz + 1] - zarr[iz]);

    const auto idx = [ny, nz](int i, int j, int k) {
        return i * ny * nz + j * nz + k;
    };

    const real_t c000 = values[idx(ix,     iy,     iz    )];
    const real_t c100 = values[idx(ix + 1, iy,     iz    )];
    const real_t c010 = values[idx(ix,     iy + 1, iz    )];
    const real_t c110 = values[idx(ix + 1, iy + 1, iz    )];
    const real_t c001 = values[idx(ix,     iy,     iz + 1)];
    const real_t c101 = values[idx(ix + 1, iy,     iz + 1)];
    const real_t c011 = values[idx(ix,     iy + 1, iz + 1)];
    const real_t c111 = values[idx(ix + 1, iy + 1, iz + 1)];

    const real_t vx00 = (1 - wx) * c000 + wx * c100;
    const real_t vx10 = (1 - wx) * c010 + wx * c110;
    const real_t vx01 = (1 - wx) * c001 + wx * c101;
    const real_t vx11 = (1 - wx) * c011 + wx * c111;
    const real_t vxy0 = (1 - wy) * vx00 + wy * vx10;
    const real_t vxy1 = (1 - wy) * vx01 + wy * vx11;
    return (1 - wz) * vxy0 + wz * vxy1;
}

inline real_t vs2vp(real_t vs) {
    return 0.9409 + 2.0947*vs - 0.8206*vs*vs + 0.2683*vs*vs*vs - 0.0251*vs*vs*vs*vs;
}

template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> vs2vp(
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& vs
) {
    return (0.9409 + 2.0947*vs.array() - 0.8206*vs.array().square() +
            0.2683*vs.array().cube() - 0.0251*vs.array().pow(4)).matrix();
}

// d(rho)/d(vp): derivative of vp2rho w.r.t. vp (scalar)
inline real_t drho_dalpha(real_t vp) {
    return 1.6612 - 2*0.4721*vp + 3*0.0671*vp*vp - 4*0.0043*vp*vp*vp + 5*0.000106*vp*vp*vp*vp;
}

// d(rho)/d(vp): element-wise, vector overload
template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> drho_dalpha(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& vp) {
    auto a = vp.array();
    return (1.6612 - 2*0.4721*a + 3*0.0671*a.square() - 4*0.0043*a.cube() + 5*0.000106*a.pow(4)).matrix();
}

// d(vp)/d(vs): derivative of vs2vp w.r.t. vs (scalar)
inline real_t dalpha_dbeta(real_t vs) {
    return 2.0947 - 2*0.8206*vs + 3*0.2683*vs*vs - 4*0.0251*vs*vs*vs;
}

// d(vp)/d(vs): element-wise, vector overload
template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> dalpha_dbeta(
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& vs
) {
    auto a = vs.array();
    return (2.0947 - 2*0.8206*a + 3*0.2683*a.square() - 4*0.0251*a.cube()).matrix();
}

inline real_t vp2rho(real_t vp) {
    return 1.6612*vp - 0.4721*vp*vp + 0.0671*vp*vp*vp - 0.0043*vp*vp*vp*vp + 0.000106*vp*vp*vp*vp*vp;
}

template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> vp2rho(
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& vp
) {
    return (1.6612*vp.array() - 0.4721*vp.array().square() + 
            0.0671*vp.array().cube() - 0.0043*vp.array().pow(4) + 
            0.000106*vp.array().pow(5)).matrix();
}

// Tensor overloads (Brocher 2005) for 3-D model fields
template<int Rank, int Options>
inline Eigen::Tensor<real_t, Rank, Options>
vs2vp(const Eigen::Tensor<real_t, Rank, Options> &vs) {
    return vs.constant(0.9409)
         + vs.constant(2.0947) * vs
         - vs.constant(0.8206) * vs * vs
         + vs.constant(0.2683) * vs * vs * vs
         - vs.constant(0.0251) * vs * vs * vs * vs;
}

template<int Rank, int Options>
inline Eigen::Tensor<real_t, Rank, Options>
vp2rho(const Eigen::Tensor<real_t, Rank, Options> &vp) {
    return vp.constant(1.6612) * vp
         - vp.constant(0.4721) * vp * vp
         + vp.constant(0.0671) * vp * vp * vp
         - vp.constant(0.0043) * vp * vp * vp * vp
         + vp.constant(0.000106) * vp * vp * vp * vp * vp;
}

// Gaussian smoothing along depth axis z with standard deviation sigma.
// Only samples within 3*sigma are included (matches the Fortran nx = sigma3/dz window).
template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> gaussian_smooth_1d(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& data,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& z,
    T sigma)
{
    const int n = static_cast<int>(data.size());
    const T dz = z(1) - z(0);
    const T sigma_sq = sigma * sigma;
    const int nx = static_cast<int>(3.0 * sigma / dz);

    Eigen::Matrix<T, Eigen::Dynamic, 1> smdata(n);
    for (int i = 0; i < n; ++i) {
        int n1 = std::max(0, i - nx);
        int n2 = std::min(n - 1, i + nx);
        T wsum = T(0), vsum = T(0);
        for (int j = n1; j <= n2; ++j) {
            T dz_ij = (j - i) * dz;
            T w = std::exp(-dz_ij * dz_ij / (T(2) * sigma_sq));
            wsum += w;
            vsum += w * data(j);
        }
        smdata(i) = vsum / wsum;
    }
    return smdata;
}

template<typename T>
inline T gps2dist(T lat0, T lon0, T lat1, T lon1)
{
    using std::sin; using std::cos; using std::atan2; using std::sqrt;

    T dlat = (lat0 - lat1) * DEG2RAD;
    T dlon = (lon1 - lon0) * DEG2RAD;
    T a = sin(dlat * T(0.5)) * sin(dlat * T(0.5))
        + sin(dlon * T(0.5)) * sin(dlon * T(0.5))
        * cos(lat0 * DEG2RAD) * cos(lat1 * DEG2RAD);
    T rad = T(2.0) * atan2(sqrt(a), sqrt(T(1.0) - a));

    return R_EARTH * rad;
}

// ---------------------------------------------------------------------------
// meshgrid
//
// Replicates NumPy's meshgrid(x, y, indexing='xy') behaviour.
// Given a 1-D x-vector (length nx) and y-vector (length ny), fills:
//   X(i,j) = x(j)   — broadcast x across rows
//   Y(i,j) = y(i)   — broadcast y across columns
// so that X and Y have shape (ny, nx), matching the NumPy default.
//
// Example (indexing='ij' style, shape (nx,ny)):
//   auto [X, Y] = meshgrid_ij(lon, lat);   // X(i,j)=lon(i), Y(i,j)=lat(j)
// ---------------------------------------------------------------------------
inline std::pair<Eigen::MatrixX<real_t>, Eigen::MatrixX<real_t>>
meshgrid(const Eigen::VectorX<real_t>& x, const Eigen::VectorX<real_t>& y)
{
    const int nx = static_cast<int>(x.size());
    const int ny = static_cast<int>(y.size());
    // shape: (ny, nx)  — same as np.meshgrid(x, y)
    Eigen::MatrixX<real_t> X = x.transpose().replicate(ny, 1);  // (ny, nx)
    Eigen::MatrixX<real_t> Y = y.replicate(1, nx);               // (ny, nx)
    return {X, Y};
}

// 'ij' indexing variant: shape (nx, ny), X(i,j)=x(i), Y(i,j)=y(j)
inline std::pair<Eigen::MatrixX<real_t>, Eigen::MatrixX<real_t>>
meshgrid_ij(const Eigen::VectorX<real_t>& x, const Eigen::VectorX<real_t>& y)
{
    const int nx = static_cast<int>(x.size());
    const int ny = static_cast<int>(y.size());
    Eigen::MatrixX<real_t> X = x.replicate(1, ny);               // (nx, ny)
    Eigen::MatrixX<real_t> Y = y.transpose().replicate(nx, 1);   // (nx, ny)
    return {X, Y};
}

// ---------------------------------------------------------------------------
// interp2d
//
// Eigen-native version of bilinear_interpolation.
//
// Interpolates a scalar field z(nx, ny) defined on a regular or irregular
// grid (xgrid, ygrid) at query points (xq, yq).
//
// Parameters:
//   xgrid – 1-D x-axis nodes, strictly increasing, length nx
//   ygrid – 1-D y-axis nodes, strictly increasing, length ny
//   z     – field values, shape (nx, ny),  z(i,j) at (xgrid(i), ygrid(j))
//   xq    – query x-coordinates, length nq
//   yq    – query y-coordinates, length nq  (same length as xq)
//
// Returns:
//   zq    – interpolated values at (xq(k), yq(k)), length nq.
//           Points outside the grid extent return NaN.
//
// Implementation:
//   Uses std::lower_bound per query point (O(log n)).  The weight
//   computation is identical to the scalar bilinear_interpolation above.
// ---------------------------------------------------------------------------
inline Eigen::VectorX<real_t> interp2d(
    const Eigen::VectorX<real_t>& xgrid,
    const Eigen::VectorX<real_t>& ygrid,
    const Eigen::MatrixX<real_t>& z,
    const Eigen::VectorX<real_t>& xq,
    const Eigen::VectorX<real_t>& yq)
{
    const int nx = static_cast<int>(xgrid.size());
    const int ny = static_cast<int>(ygrid.size());
    const int nq = static_cast<int>(xq.size());

    Eigen::VectorX<real_t> zq(nq);

    const real_t* px = xgrid.data();
    const real_t* py = ygrid.data();

    for (int k = 0; k < nq; ++k) {
        const real_t qx = xq(k);
        const real_t qy = yq(k);

        // --- locate bounding cell in x ---
        if (qx < xgrid(0) || qx > xgrid(nx - 1) ||
            qy < ygrid(0) || qy > ygrid(ny - 1)) {
            zq(k) = std::numeric_limits<real_t>::quiet_NaN();
            continue;
        }

        int i = static_cast<int>(
            std::lower_bound(px, px + nx, qx) - px);
        if (i == nx) --i;
        if (i > 0 && xgrid(i) > qx) --i;
        i = std::clamp(i, 0, nx - 2);

        int j = static_cast<int>(
            std::lower_bound(py, py + ny, qy) - py);
        if (j == ny) --j;
        if (j > 0 && ygrid(j) > qy) --j;
        j = std::clamp(j, 0, ny - 2);

        // --- bilinear weights ---
        const real_t x1 = xgrid(i),  x2 = xgrid(i + 1);
        const real_t y1 = ygrid(j),  y2 = ygrid(j + 1);
        const real_t tx = (qx - x1) / (x2 - x1);
        const real_t ty = (qy - y1) / (y2 - y1);

        zq(k) = (1 - tx) * (1 - ty) * z(i,     j    )
              + (    tx) * (1 - ty) * z(i + 1, j    )
              + (1 - tx) * (    ty) * z(i,     j + 1)
              + (    tx) * (    ty) * z(i + 1, j + 1);
    }
    return zq;
}

// 2-D overload: xq and yq are matrices (e.g. from meshgrid / meshgrid_ij).
// Returns a matrix of the same shape as xq/yq.
inline Eigen::MatrixX<real_t> interp2d(
    const Eigen::VectorX<real_t>& xgrid,
    const Eigen::VectorX<real_t>& ygrid,
    const Eigen::MatrixX<real_t>& z,
    const Eigen::MatrixX<real_t>& xq,
    const Eigen::MatrixX<real_t>& yq)
{
    const int rows = static_cast<int>(xq.rows());
    const int cols = static_cast<int>(xq.cols());

    // Flatten to column vectors — materialize to VectorX so the 1-D overload
    // is selected unambiguously (Map converts to both VectorX and MatrixX).
    Eigen::VectorX<real_t> xq_flat =
        Eigen::Map<const Eigen::VectorX<real_t>>(xq.data(), rows * cols);
    Eigen::VectorX<real_t> yq_flat =
        Eigen::Map<const Eigen::VectorX<real_t>>(yq.data(), rows * cols);

    Eigen::VectorX<real_t> zq_flat = interp2d(xgrid, ygrid, z, xq_flat, yq_flat);

    // Reshape back to (rows, cols)
    return Eigen::Map<Eigen::MatrixX<real_t>>(zq_flat.data(), rows, cols);
}

// ---------------------------------------------------------------------------
// interp1d
//
// Linear interpolation of a 1-D field y defined on grid xgrid at query
// points xq.
//
// Parameters:
//   xgrid – 1-D x-axis nodes, strictly increasing, length nx
//   y     – field values, length nx
//   xq    – query x-coordinates, length nq
//
// Returns:
//   yq    – interpolated values at xq(k), length nq.
//           Points outside the grid extent return NaN.
// ---------------------------------------------------------------------------
inline Eigen::VectorX<real_t> interp1d(
    const Eigen::VectorX<real_t>& xgrid,
    const Eigen::VectorX<real_t>& y,
    const Eigen::VectorX<real_t>& xq)
{
    const int nx = static_cast<int>(xgrid.size());
    const int nq = static_cast<int>(xq.size());
    const real_t* px = xgrid.data();

    Eigen::VectorX<real_t> yq(nq);

    for (int k = 0; k < nq; ++k) {
        const real_t qx = xq(k);

        // Extrapolate at boundaries using constant extension (nearer endpoint value)
        if (qx < xgrid(0)) {
            yq(k) = y(0);
            continue;
        }
        if (qx > xgrid(nx - 1)) {
            yq(k) = y(nx - 1);
            continue;
        }

        int i = static_cast<int>(
            std::lower_bound(px, px + nx, qx) - px);
        if (i == nx) --i;
        if (i > 0 && xgrid(i) > qx) --i;
        i = std::clamp(i, 0, nx - 2);

        const real_t x1 = xgrid(i), x2 = xgrid(i + 1);
        const real_t t = (qx - x1) / (x2 - x1);
        yq(k) = (1 - t) * y(i) + t * y(i + 1);
    }
    return yq;
}

// ---------------------------------------------------------------------------
// gaussian_smooth_geo_2
//
// 2-D Gaussian smoothing on a geographic (lon/lat) grid.
// sigma is the half-width in arc-degrees (~111*sigma km).
//
// Window logic (faithfully matches the Fortran original):
//   hx = ceil(3*sigma / dx)          — lon half-width, constant
//   hy = ceil(3*sigma / cos(lat) / dy) — lat half-width, varies with lat
//
// Vectorization strategy:
//   • sin²(Δlat/2) and cos(lat_sub) are precomputed once per latitude row j
//     as Eigen arrays (wn elements).
//   • For each i, sin²(Δlon/2) is an Eigen array (wm elements).
//   • The haversine 'a' matrix (wm×wn) is built with two outer-product ops,
//     so the weight matrix w is computed without any scalar loop.
// ---------------------------------------------------------------------------
inline Eigen::MatrixX<real_t> gaussian_smooth_geo_2(
    const Eigen::MatrixX<real_t>& data,
    const Eigen::VectorX<real_t>& lons,
    const Eigen::VectorX<real_t>& lats,
    real_t sigma)
{
    const int nx = data.rows();   // n_lon
    const int ny = data.cols();   // n_lat

    const real_t sigma3   = 3.0 * sigma;
    const real_t sigma_sq = sigma * sigma;
    const real_t dx = lons(1) - lons(0);   // lon spacing [deg]
    const real_t dy = lats(1) - lats(0);   // lat spacing [deg]

    // lon window half-width (constant across latitudes — matches Fortran nx)
    const int hx = static_cast<int>(sigma3 / dx);

    Eigen::MatrixX<real_t> smdata = Eigen::MatrixX<real_t>::Zero(nx, ny);

    for (int j = 0; j < ny; ++j) {
        const real_t lat0_rad = lats(j) * DEG2RAD;
        const real_t cos_lat0 = std::cos(lat0_rad);

        // lat window half-width (varies — matches Fortran ny = sigma3/cosd/dy)
        const int hy = static_cast<int>(sigma3 / cos_lat0 / dy);
        const int n1 = std::max(0, j - hy);
        const int n2 = std::min(ny - 1, j + hy);
        const int wn = n2 - n1 + 1;

        // --- precompute haversine lat-terms for all wn lat points (vectorized) ---
        // sin²(Δlat/2)  (wn,)
        Eigen::ArrayX<real_t> sin2_dlat =
            ((lats.segment(n1, wn).array() - lats(j)) * (DEG2RAD * 0.5)).sin().square();
        // cos(lat_sub)  (wn,)
        Eigen::ArrayX<real_t> cos_lat1 =
            (lats.segment(n1, wn).array() * DEG2RAD).cos();

        for (int i = 0; i < nx; ++i) {
            const int m1 = std::max(0, i - hx);
            const int m2 = std::min(nx - 1, i + hx);
            const int wm = m2 - m1 + 1;

            // sin²(Δlon/2)  (wm,)
            Eigen::ArrayX<real_t> sin2_dlon =
                ((lons.segment(m1, wm).array() - lons(i)) * (DEG2RAD * 0.5)).sin().square();

            // Haversine 'a' matrix  (wm × wn):
            //   a(mi,ni) = sin2_dlat(ni)
            //            + cos_lat0 * cos_lat1(ni) * sin2_dlon(mi)
            // Built as two outer products (no scalar loop):
            Eigen::ArrayX<real_t> lon_contrib = cos_lat0 * sin2_dlon; // (wm,)
            Eigen::ArrayXX<real_t> a =
                sin2_dlat.transpose().replicate(wm, 1)                          // (wm,wn): each row = sin2_dlat
                + (lon_contrib.matrix() * cos_lat1.matrix().transpose()).array(); // (wm,wn): outer product

            // delta [arc-deg] = 2 * RAD2DEG * atan(sqrt(a / (1-a)))
            // w = exp(-delta² / (2*sigma²))
            // Combined: w = exp(-(2*RAD2DEG)² * atan²(√(a/(1-a))) / (2*sigma²))
            a = a.min(static_cast<real_t>(1.0));  // guard numerical noise near 1
            const real_t inv_2sig2 = 1.0 / (2.0 * sigma_sq);
            constexpr real_t two_rad2deg = 2.0 * RAD2DEG;
            Eigen::ArrayXX<real_t> w =
                (-(two_rad2deg * (a / (1.0 - a)).sqrt().atan()).square() * inv_2sig2).exp();

            smdata(i, j) = (w * data.block(m1, n1, wm, wn).array()).sum() / w.sum();
        }
    }
    return smdata;
}

// ---------------------------------------------------------------------------
// gradient_2_geo
//
// Computes the geographic gradient of a 2-D field f(nx, ny) defined on a
// regular lon/lat grid.  The output arrays tx and ty hold the x (longitude)
// and y (latitude) components of the gradient in units of [f-unit / km].
//
// Central differences are used for interior points; one-sided differences
// at the boundaries.  Grid spacing is converted to km via the spherical-
// Earth approximation:
//     dx_km = R_EARTH * dlon_deg * DEG2RAD * cos(lat_deg * DEG2RAD)
//     dy_km = R_EARTH * dlat_deg * DEG2RAD
//
// Parameters:
//   f   – field matrix, shape (nx, ny)
//   lon – longitude vector, length nx  [degrees]
//   lat – latitude  vector, length ny  [degrees]
//   tx  – output: gradient in x/lon direction, shape (nx, ny)
//   ty  – output: gradient in y/lat direction, shape (nx, ny)
// ---------------------------------------------------------------------------
inline void gradient_2_geo(
    const Eigen::MatrixX<real_t>& f,
    const Eigen::VectorX<real_t>& lon,
    const Eigen::VectorX<real_t>& lat,
    Eigen::MatrixX<real_t>& tx,
    Eigen::MatrixX<real_t>& ty)
{
    const int nx = static_cast<int>(f.rows());
    const int ny = static_cast<int>(f.cols());

    // --- grid spacing vectors (km) -------------------------------------------
    // dx1(i) = lon[i]-lon[i-1],  dx1(0) = 0  (boundary sentinel, never divided)
    // dy1(j) = lat[j]-lat[j-1],  dy1(0) = 0
    Eigen::VectorX<real_t> dx1 = Eigen::VectorX<real_t>::Zero(nx);
    Eigen::VectorX<real_t> dy1 = Eigen::VectorX<real_t>::Zero(ny);
    dx1.tail(nx - 1) = lon.tail(nx - 1) - lon.head(nx - 1);
    dy1.tail(ny - 1) = lat.tail(ny - 1) - lat.head(ny - 1);

    // Convert to km.  dx additionally scaled by cos(lat) per column.
    // cos_lat(j) shape: (ny,)
    Eigen::VectorX<real_t> cos_lat = (lat.array() * DEG2RAD).cos();
    dx1 *= R_EARTH * DEG2RAD;
    dy1 *= R_EARTH * DEG2RAD;

    // dx(i,j) = dx1_km(i) * cos_lat(j)  →  outer product (nx×1) × (1×ny)
    // dy(i,j) = dy1_km(j)               →  broadcast (1×ny) over nx rows
    Eigen::MatrixX<real_t> dx = dx1 * cos_lat.transpose();          // (nx, ny)
    Eigen::MatrixX<real_t> dy = Eigen::VectorX<real_t>::Ones(nx)
                                * dy1.transpose();                   // (nx, ny)

    tx.resize(nx, ny);
    ty.resize(nx, ny);

    // --- x-direction gradient (along rows) -----------------------------------
    // Interior: central difference  tx(i,j) = (f(i+1,j)-f(i-1,j)) / (2*dx(i,j))
    //   f.bottomRows(nx-2) = rows 2..nx-1  →  f(i+1,:) for i=1..nx-2
    //   f.topRows(nx-2)    = rows 0..nx-3  →  f(i-1,:) for i=1..nx-2
    tx.middleRows(1, nx - 2).array() =
        (f.bottomRows(nx - 2) - f.topRows(nx - 2)).array()
        / (2.0 * dx.middleRows(1, nx - 2).array());
    // Boundaries: one-sided
    tx.row(0).array()      = (f.row(1)      - f.row(0)     ).array() / dx.row(1).array();
    tx.row(nx-1).array()   = (f.row(nx-1)   - f.row(nx-2)  ).array() / dx.row(nx-1).array();

    // --- y-direction gradient (along columns) --------------------------------
    // Interior: central difference  ty(i,j) = (f(i,j+1)-f(i,j-1)) / (2*dy(i,j))
    //   f.rightCols(ny-2) = cols 2..ny-1  →  f(:,j+1) for j=1..ny-2
    //   f.leftCols(ny-2)  = cols 0..ny-3  →  f(:,j-1) for j=1..ny-2
    ty.middleCols(1, ny - 2).array() =
        (f.rightCols(ny - 2) - f.leftCols(ny - 2)).array()
        / (2.0 * dy.middleCols(1, ny - 2).array());
    // Boundaries: one-sided
    ty.col(0).array()      = (f.col(1)      - f.col(0)     ).array() / dy.col(1).array();
    ty.col(ny-1).array()   = (f.col(ny-1)   - f.col(ny-2)  ).array() / dy.col(ny-1).array();
}

inline Eigen::VectorX<real_t> extract_1d_from_3d(
    const real_t* data, const int ix, const int iy, const int nz
) {
    Eigen::VectorX<real_t> data_1d(nz);
    for (int k = 0; k < nz; k++){
        data_1d(k) = data[I2V(ix, iy, k)];
    }
    return data_1d;
}

// ---------------------------------------------------------------------------
// Convert a numeric array into fixed-precision string values for CSV output.
// rapidcsv writing here is string-based to keep formatting explicit/consistent.
inline std::vector<std::string> fmt_col(const real_t* data, int n, int prec = 6) {
    std::vector<std::string> v(n);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec);
    for (int i = 0; i < n; ++i) {
        oss.str(""); oss.clear();
        oss << data[i];
        v[i] = oss.str();
    }
    return v;
}

// ---------------------------------------------------------------------------
// Radial anisotropy parameter conversions
//
// Converts between (vsv, vsh) and (Vs, zeta) parameterizations.
//
// Definitions:
//   Vs   = sqrt((2*vsv + vsh) / 3)  — RMS S-wave velocity
//   zeta = vsh^2 / vsv^2             — anisotropy ratio
//
// Supports scalar, vector (Eigen::VectorX), and tensor (Eigen::Tensor<3>) inputs.
// ---------------------------------------------------------------------------

// Scalar version: compute Vs from vsv and vsh
inline real_t vsvvsh2vs(real_t vsv, real_t vsh) {
    return std::sqrt((2 * vsv* vsv + vsh * vsh) / 3);
}

// Scalar version: compute zeta from vsv and vsh
inline real_t vsvvsh2zeta(real_t vsv, real_t vsh) {
    return (vsh * vsh) / (vsv * vsv);
}

// Scalar version: recover vsv and vsh from Vs and zeta
// Given: Vs = sqrt((2*vsv^2 + vsh^2) / 3), zeta = vsh^2 / vsv^2
// Solution: vsv = sqrt(3*Vs^2 / (2 + zeta)), vsh = vsv * sqrt(zeta)
inline std::pair<real_t, real_t> recover_anisotropy(real_t Vs, real_t zeta) {
    real_t sqrt_zeta = std::sqrt(zeta);
    real_t vsv = std::sqrt(3 * Vs * Vs / (2 + zeta));
    real_t vsh = vsv * sqrt_zeta;
    return {vsv, vsh};
}

// Vector version: compute Vs from vsv and vsh vectors (Ref overload)
template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> vsvvsh2vs(
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& vsv,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& vsh
) {
    return (
        (T(2) * vsv.array().square() + vsh.array().square()) / T(3)
    ).sqrt().matrix();
}

// Vector version: compute Vs from vsv and vsh vectors (Matrix overload)
template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> vsvvsh2vs(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& vsv,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& vsh
) {
    return (
        (T(2) * vsv.array().square() + vsh.array().square()) / T(3)
    ).sqrt().matrix();
}

// Vector version: compute zeta from vsv and vsh vectors (Ref overload)
template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> vsvvsh2zeta(
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& vsv,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& vsh
) {
    return (vsh.array().square() / vsv.array().square()).matrix();
}

// Vector version: compute zeta from vsv and vsh vectors (Matrix overload)
template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> vsvvsh2zeta(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& vsv,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& vsh
) {
    return (vsh.array().square() / vsv.array().square()).matrix();
}

// Tensor version: compute Vs from vsv and vsh tensors
template<int Rank, int Options>
inline Eigen::Tensor<real_t, Rank, Options> vsvvsh2vs(
    const Eigen::Tensor<real_t, Rank, Options>& vsv,
    const Eigen::Tensor<real_t, Rank, Options>& vsh
) {
    return (
        (vsv.constant(2.0) * vsv * vsv + vsh * vsh) / vsv.constant(3.0)
    ).sqrt();
}

// Tensor version: compute zeta from vsv and vsh tensors
template<int Rank, int Options>
inline Eigen::Tensor<real_t, Rank, Options> vsvvsh2zeta(
    const Eigen::Tensor<real_t, Rank, Options>& vsv,
    const Eigen::Tensor<real_t, Rank, Options>& vsh
) {
    return vsh * vsh / (vsv * vsv);
}
