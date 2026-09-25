#pragma once

#include <vector>
#include <stdexcept>
#include <string>
#include "config.h"

constexpr double TWOPI = 2.0 * PI;
constexpr int MAX_LAYERS = 512;

struct Layer {
    float thickness;  /* km  -- set to 0 for the halfspace (last layer) */
    float vp;         /* km/s */
    float vs;         /* km/s */
    float density;    /* g/cm^3 */
};

struct LoveEigenResult {
    double cp;                   // phase velocity (km/s)
    double cg;                   // group velocity (km/s)
    std::vector<double> disp;   // displacement eigenfunction [nlayer]
    std::vector<double> stress; // stress eigenfunction [nlayer]
    std::vector<double> dc2db;  // dc/dVs [nlayer]
    std::vector<double> dc2dh;  // dc/dh  [nlayer]
    std::vector<double> dc2dr;  // dc/drho [nlayer]
};

struct LoveGroupKernelResult : LoveEigenResult {
    std::vector<double> du2db;  // dU/dVs [nlayer]
    std::vector<double> du2dh;  // dU/dh  [nlayer]
    std::vector<double> du2dr;  // dU/drho [nlayer]
};


std::vector<double> disper(const float *thkm, const float *vpm, const float *vsm,
                             const float *rhom, int nlayer, int iflsph, int iwave,
                             int mode, int igr, int kmax, const double *t);

// Diagnostics of fundamental-mode failures in disper() (per thread).
// disper() writes a rate-limited report to stderr on every failure; callers
// that want to attach context can reset, run, then query.
void disper_diag_reset();
int disper_diag_nfail();                    // failures since the last reset
const std::string &disper_diag_report();    // full report of the first one

// Love-wave eigenfunctions + phase-velocity kernels at one period.
// phase_vel_km_s: phase velocity at period_s (from surfdisp::dispersion)
LoveEigenResult lovePhaseKernel(
    const std::vector<Layer>& model,
    double period_s,
    double phase_vel_km_s,
    EarthModel earth = EarthModel::Spherical
);

// Love-wave phase + group velocity kernels.
// t/cp: the target period; t1/cp1, t2/cp2: bracket for finite-difference group kernel
LoveGroupKernelResult loveGroupKernel(
    const std::vector<Layer>& model,
    double t,  double cp,
    double t1, double cp1,
    double t2, double cp2,
    EarthModel earth = EarthModel::Spherical
);

// Array-based overload: thk/vs/rho are flat float arrays of length nlayer.
LoveEigenResult lovePhaseKernel(
    int nlayer,
    const float* thk, const float* vs, const float* rho,
    double period_s,
    double phase_vel_km_s,
    EarthModel earth = EarthModel::Spherical
);

// Array-based overload
LoveGroupKernelResult loveGroupKernel(
    int nlayer,
    const float* thk, const float* vs, const float* rho,
    double t,  double cp,
    double t1, double cp1,
    double t2, double cp2,
    EarthModel earth = EarthModel::Spherical
);

struct RayleighEigenResult {
    double cp, cg;
    std::vector<double> dispu;   // ur: horizontal displacement [nlayer]
    std::vector<double> dispw;   // uz: vertical displacement [nlayer]
    std::vector<double> stressu; // tr: horizontal stress [nlayer]
    std::vector<double> stressw; // tz: vertical stress [nlayer]
    std::vector<double> dc2da;   // dc/dVp [nlayer]
    std::vector<double> dc2db;   // dc/dVs [nlayer]
    std::vector<double> dc2dh;   // dc/dh  [nlayer]
    std::vector<double> dc2dr;   // dc/drho [nlayer]
};

struct HTIResult : RayleighEigenResult {
    std::vector<double> dc2dgc;  // dc/dgc [nlayer]
    std::vector<double> dc2dgs;  // dc/dgs [nlayer]
};

struct RayleighGroupKernelResult : RayleighEigenResult {
    std::vector<double> du2da, du2db, du2dh, du2dr;
};

// vector<Layer> overloads
RayleighEigenResult rayleighPhaseKernel(const std::vector<Layer>& model,
    double period_s, double phase_vel_km_s,
    EarthModel earth = EarthModel::Flat);

HTIResult rayleighPhaseKernel_hti(const std::vector<Layer>& model,
    double period_s, double phase_vel_km_s,
    EarthModel earth = EarthModel::Flat);

RayleighGroupKernelResult rayleighGroupKernel(const std::vector<Layer>& model,
    double t, double cp, double t1, double cp1, double t2, double cp2,
    EarthModel earth = EarthModel::Flat);

// Array overloads
RayleighEigenResult rayleighPhaseKernel(int nlayer,
    const float* thk, const float* vp, const float* vs, const float* rho,
    double period_s, double phase_vel_km_s,
    EarthModel earth = EarthModel::Flat);

HTIResult rayleighPhaseKernel_hti(int nlayer,
    const float* thk, const float* vp, const float* vs, const float* rho,
    double period_s, double phase_vel_km_s,
    EarthModel earth = EarthModel::Flat);

RayleighGroupKernelResult rayleighGroupKernel(int nlayer,
    const float* thk, const float* vp, const float* vs, const float* rho,
    double t, double cp, double t1, double cp1, double t2, double cp2,
    EarthModel earth = EarthModel::Flat);
