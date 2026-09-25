/*
 * Original: D. R. Russell, R. B. Herrmann (1986, 1991)
 * Modified: Huajian Yao (2010), Nanqiao Du (2019, 2023)
 * Modified from Fortran to C++ and adapted for SurfATTPP by Mijian Xu (2026)
 *
 */

#include "surfdisp.hpp"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>


/* ------------------------------------------------------------------ */
/* Failure diagnostics                                                  */
/* ------------------------------------------------------------------ */

/* What getsol() did while bracketing a root (filled for diagnostics). */
struct GetsolTrace {
    enum Reason { OK = 0, ROOT_ABOVE_BETMX, PASSED_BETMX, BELOW_CM, STEP_CAP };
    double c_start = 0.0;   /* phase velocity where the bracket search started */
    double del_start = 0.0; /* period-equation value at c_start */
    int    idir    = 0;     /* initial search direction (+1 up, -1 down) */
    long   nstep   = 0;     /* number of dc steps taken */
    double c_end   = 0.0;   /* last velocity tested, or the root found */
    int    reason  = OK;
};

static const char *getsol_reason_str(int reason)
{
    switch (reason) {
    case GetsolTrace::OK:               return "ok";
    case GetsolTrace::ROOT_ABOVE_BETMX: return "root found but above betmx (max Vs of the column)";
    case GetsolTrace::PASSED_BETMX:     return "no sign change of the period equation up to betmx+dc";
    case GetsolTrace::BELOW_CM:         return "search went below the lower bound cm";
    case GetsolTrace::STEP_CAP:         return "step limit reached (non-finite period equation?)";
    default:                            return "unknown";
    }
}

/* State since the last disper_diag_reset() on this thread. */
static thread_local int         g_nfail_since_reset = 0;
static thread_local std::string g_first_report;
/* Per-process count used to rate-limit the stderr output. */
static long g_nfail_total = 0;
constexpr long MAX_FULL_REPORTS  = 3;    /* full reports written to stderr */
constexpr long MAX_SHORT_REPORTS = 100;  /* one-line reports after that */

void disper_diag_reset()
{
    g_nfail_since_reset = 0;
    g_first_report.clear();
}

int disper_diag_nfail() { return g_nfail_since_reset; }

const std::string &disper_diag_report() { return g_first_report; }

static void record_failure(const std::string &summary, const std::string &report)
{
    ++g_nfail_since_reset;
    if (g_first_report.empty()) g_first_report = report;

    ++g_nfail_total;
    if (g_nfail_total <= MAX_FULL_REPORTS) {
        fprintf(stderr, "%s", report.c_str());
    } else if (g_nfail_total <= MAX_FULL_REPORTS + MAX_SHORT_REPORTS) {
        fprintf(stderr, "%s\n", summary.c_str());
    } else if (g_nfail_total == MAX_FULL_REPORTS + MAX_SHORT_REPORTS + 1) {
        fprintf(stderr, "WARNING: disper: further failure reports on this process are suppressed\n");
    }
    fflush(stderr);
}


/* ------------------------------------------------------------------ */
/* Forward declarations                                                 */
/* ------------------------------------------------------------------ */

static void gtsolh(float a, float b, float &c);

static void getsol(double t1, double &c1, double clow, double dc, double cm,
                   double betmx, int &iret, int ifunc, int ifirst,
                   float *d, float *a, float *b, float *rho,
                   float *rtp, float *dtp, float *btp, int mmax, int llw,
                   double &del1st, GetsolTrace *trace = nullptr);

static void sphere_func(int ifunc, int iflag,
                        float *d, float *a, float *b, float *rho,
                        float *rtp, float *dtp, float *btp,
                        int mmax, int llw, float &dhalf);

static void nevill(double t, double c1, double c2, double del1, double del2,
                   int ifunc, double &cc,
                   float *d, float *a, float *b, float *rho,
                   float *rtp, float *dtp, float *btp, int mmax, int llw);

static void half_func(double c1, double c2, double &c3, double &del3,
                      double omega, int ifunc,
                      float *d, float *a, float *b, float *rho,
                      float *rtp, float *dtp, float *btp, int mmax, int llw);

static double dltar(double wvno, double omega, int kk,
                    float *d, float *a, float *b, float *rho,
                    float *rtp, float *dtp, float *btp, int mmax, int llw);

static double dltar1(double wvno, double omega,
                     float *d, float *a, float *b, float *rho,
                     float *rtp, float *dtp, float *btp, int mmax, int llw);

static double dltar4(double wvno, double omega,
                     float *d, float *a, float *b, float *rho,
                     float *rtp, float *dtp, float *btp, int mmax, int llw);

static void var_func(double p, double q, double ra, double rb,
                     double wvno, double xka, double xkb, double dpth,
                     double &w, double &cosp, double &exa,
                     double &a0, double &cpcq, double &cpy, double &cpz,
                     double &cqw, double &cqx, double &xy, double &xz,
                     double &wy, double &wz);

static void normc(double ee[5], double &ex);

static void dnka(double ca[5][5], double wvno2, double gam, double gammk,
                 double rho_val,
                 double a0, double cpcq, double cpy, double cpz,
                 double cqw, double cqx, double xy, double xz,
                 double wy, double wz);

/* ------------------------------------------------------------------ */
/* gtsolh: Newton's iteration for halfspace Rayleigh starting velocity */
/* ------------------------------------------------------------------ */
static void gtsolh(float a, float b, float &c)
{
    c = 0.95f * b;
    for (int i = 0; i < 5; i++) {
        float gamma  = b / a;
        float kappa  = c / b;
        float k2     = kappa * kappa;
        float gk2    = (gamma * kappa) * (gamma * kappa);
        float fac1   = sqrtf(1.0f - gk2);
        float fac2   = sqrtf(1.0f - k2);
        float fr     = (2.0f - k2) * (2.0f - k2) - 4.0f * fac1 * fac2;
        float frp    = -4.0f * (2.0f - k2) * kappa
                       + 4.0f * fac2 * gamma * gamma * kappa / fac1
                       + 4.0f * fac1 * kappa / fac2;
        frp /= b;
        c -= fr / frp;
    }
}

/* ------------------------------------------------------------------ */
/* sphere_func: spherical-to-flat earth transformation                 */
/* ------------------------------------------------------------------ */
static void sphere_func(int ifunc, int iflag,
                        float *d, float *a, float *b, float *rho,
                        float *rtp, float *dtp, float *btp,
                        int mmax, int /*llw*/, float &dhalf)
{
    const double ar = 6370.0;
    double dr = 0.0;
    double r0 = ar;

    d[mmax - 1] = 1.0f;

    if (iflag == 0) {
        for (int i = 0; i < mmax; i++) {
            dtp[i] = d[i];
            rtp[i] = rho[i];
        }
        for (int i = 0; i < mmax; i++) {
            dr += (double)d[i];
            double r1 = ar - dr;
            double z0 = ar * std::log(ar / r0);
            double z1 = ar * std::log(ar / r1);
            d[i]   = (float)(z1 - z0);
            double tmp = (ar + ar) / (r0 + r1);
            a[i]   *= (float)tmp;
            b[i]   *= (float)tmp;
            btp[i]  = (float)tmp;
            r0 = r1;
        }
        dhalf = d[mmax - 1];
    } else {
        d[mmax - 1] = dhalf;
        for (int i = 0; i < mmax; i++) {
            if (ifunc == 1)
                rho[i] = rtp[i] * std::pow(btp[i], -5.0f);
            else if (ifunc == 2)
                rho[i] = rtp[i] * std::pow(btp[i], -2.275f);
        }
    }
    d[mmax - 1] = 0.0f;
}

/* ------------------------------------------------------------------ */
/* var_func: P/S eigenfunction products for compound matrix            */
/* ------------------------------------------------------------------ */
static void var_func(double p, double q, double ra, double rb,
                     double wvno, double xka, double xkb, double dpth,
                     double &w, double &cosp, double &exa,
                     double &a0, double &cpcq, double &cpy, double &cpz,
                     double &cqw, double &cqx, double &xy, double &xz,
                     double &wy, double &wz)
{
    exa = 0.0;
    a0  = 1.0;

    double pex = 0.0, sex = 0.0;
    double x, cosq, y, z;

    /* P-wave */
    if (wvno < xka) {
        double sinp = std::sin(p);
        w    =  sinp / ra;
        x    = -ra * sinp;
        cosp =  std::cos(p);
    } else if (wvno == xka) {
        cosp = 1.0;
        w    = dpth;
        x    = 0.0;
    } else {
        pex = p;
        double fac = (p < 16.0) ? std::exp(-2.0 * p) : 0.0;
        cosp = (1.0 + fac) * 0.5;
        double sinp = (1.0 - fac) * 0.5;
        w = sinp / ra;
        x = ra * sinp;
    }

    /* S-wave */
    if (wvno < xkb) {
        double sinq = std::sin(q);
        y    =  sinq / rb;
        z    = -rb * sinq;
        cosq =  std::cos(q);
    } else if (wvno == xkb) {
        cosq = 1.0;
        y    = dpth;
        z    = 0.0;
    } else {
        sex = q;
        double fac = (q < 16.0) ? std::exp(-2.0 * q) : 0.0;
        cosq = (1.0 + fac) * 0.5;
        double sinq = (1.0 - fac) * 0.5;
        y = sinq / rb;
        z = rb * sinq;
    }

    exa  = pex + sex;
    a0   = (exa < 60.0) ? std::exp(-exa) : 0.0;
    cpcq = cosp * cosq;
    cpy  = cosp * y;
    cpz  = cosp * z;
    cqw  = cosq * w;
    cqx  = cosq * x;
    xy   = x * y;
    xz   = x * z;
    wy   = w * y;
    wz   = w * z;
    /* Trailing modification of cosq/y/z in original Fortran var is dead code */
}

/* ------------------------------------------------------------------ */
/* normc: normalize compound-matrix vector, return log of scale factor */
/* ------------------------------------------------------------------ */
static void normc(double ee[5], double &ex)
{
    double t1 = 0.0;
    for (int i = 0; i < 5; i++)
        if (std::fabs(ee[i]) > t1) t1 = std::fabs(ee[i]);
    if (t1 < 1.0e-40) t1 = 1.0;
    for (int i = 0; i < 5; i++)
        ee[i] /= t1;
    ex = std::log(t1);
}

/* ------------------------------------------------------------------ */
/* dnka: Dunkin's 5x5 compound matrix for one layer                    */
/*                                                                      */
/* Indexing: ca[i][j] in C++ == Fortran ca(i+1, j+1)                   */
/* The matrix-vector product in dltar4 is ee[i] = sum_j e[j]*ca[j][i] */
/* ------------------------------------------------------------------ */
static void dnka(double ca[5][5], double wvno2, double gam, double gammk,
                 double rho_val,
                 double a0, double cpcq, double cpy, double cpz,
                 double cqw, double cqx, double xy, double xz,
                 double wy, double wz)
{
    const double one = 1.0, two = 2.0;
    double gamm1  = gam - one;
    double twgm1  = gam + gamm1;
    double gmgmk  = gam * gammk;
    double gmgm1  = gam * gamm1;
    double gm1sq  = gamm1 * gamm1;
    double rho2   = rho_val * rho_val;
    double a0pq   = a0 - cpcq;

    ca[0][0] = cpcq - two*gmgm1*a0pq - gmgmk*xz - wvno2*gm1sq*wy;
    ca[0][1] = (wvno2*cpy - cqx) / rho_val;
    ca[0][2] = -(twgm1*a0pq + gammk*xz + wvno2*gamm1*wy) / rho_val;
    ca[0][3] = (cpz - wvno2*cqw) / rho_val;
    ca[0][4] = -(two*wvno2*a0pq + xz + wvno2*wvno2*wy) / rho2;

    ca[1][0] = (gmgmk*cpz - gm1sq*cqw) * rho_val;
    ca[1][1] = cpcq;
    ca[1][2] = gammk*cpz - gamm1*cqw;
    ca[1][3] = -wz;
    ca[1][4] = ca[0][3];   /* ca(2,5) = ca(1,4) */

    ca[3][0] = (gm1sq*cpy - gmgmk*cqx) * rho_val;
    ca[3][1] = -xy;
    ca[3][2] = gamm1*cpy - gammk*cqx;
    ca[3][3] = ca[1][1];   /* ca(4,4) = ca(2,2) */
    ca[3][4] = ca[0][1];   /* ca(4,5) = ca(1,2) */

    ca[4][0] = -(two*gmgmk*gm1sq*a0pq + gmgmk*gmgmk*xz + gm1sq*gm1sq*wy) * rho2;
    ca[4][1] = ca[3][0];   /* ca(5,2) = ca(4,1) */
    ca[4][2] = -(gammk*gamm1*twgm1*a0pq + gam*gammk*gammk*xz
                 + gamm1*gm1sq*wy) * rho_val;
    ca[4][3] = ca[1][0];   /* ca(5,4) = ca(2,1) */
    ca[4][4] = ca[0][0];   /* ca(5,5) = ca(1,1) */

    double tv = -two * wvno2;
    ca[2][0] = tv * ca[4][2];                      /* ca(3,1) = t*ca(5,3) */
    ca[2][1] = tv * ca[3][2];                      /* ca(3,2) = t*ca(4,3) */
    ca[2][2] = a0 + two * (cpcq - ca[0][0]);       /* ca(3,3) */
    ca[2][3] = tv * ca[1][2];                      /* ca(3,4) = t*ca(2,3) */
    ca[2][4] = tv * ca[0][2];                      /* ca(3,5) = t*ca(1,3) */
}

/* ------------------------------------------------------------------ */
/* dltar1: SH (Love) period equation -- Haskell-Thompson propagator    */
/* ------------------------------------------------------------------ */
static double dltar1(double wvno, double omega,
                     float *d, float *a, float *b, float *rho,
                     float * /*rtp*/, float * /*dtp*/, float * /*btp*/,
                     int mmax, int llw)
{
    double beta1 = (double)b[mmax - 1];
    double rho1  = (double)rho[mmax - 1];
    double xkb   = omega / beta1;
    double wvnop = wvno + xkb;
    double wvnom = std::fabs(wvno - xkb);
    double rb    = std::sqrt(wvnop * wvnom);
    double e1    = rho1 * rb;
    double e2    = 1.0 / (beta1 * beta1);

    for (int m = mmax - 2; m >= llw - 1; m--) {
        beta1 = (double)b[m];
        rho1  = (double)rho[m];
        double xmu = rho1 * beta1 * beta1;
        xkb   = omega / beta1;
        wvnop = wvno + xkb;
        wvnom = std::fabs(wvno - xkb);
        rb    = std::sqrt(wvnop * wvnom);
        double q = (double)d[m] * rb;

        double sinq, cosq, y, z;
        if (wvno < xkb) {
            sinq = std::sin(q);
            y    =  sinq / rb;
            z    = -rb * sinq;
            cosq =  std::cos(q);
        } else if (wvno == xkb) {
            cosq = 1.0;
            y    = (double)d[m];
            z    = 0.0;
        } else {
            double fac = (q < 16.0) ? std::exp(-2.0 * q) : 0.0;
            cosq = (1.0 + fac) * 0.5;
            sinq = (1.0 - fac) * 0.5;
            y    = sinq / rb;
            z    = rb * sinq;
        }

        double e10  = e1 * cosq + e2 * xmu * z;
        double e20  = e1 * y / xmu + e2 * cosq;
        double xnor = std::fabs(e10);
        double ynor = std::fabs(e20);
        if (ynor > xnor) xnor = ynor;
        if (xnor < 1.0e-40) xnor = 1.0;
        e1 = e10 / xnor;
        e2 = e20 / xnor;
    }
    return e1;
}

/* ------------------------------------------------------------------ */
/* dltar4: P-SV (Rayleigh) period equation -- Dunkin compound matrix   */
/* ------------------------------------------------------------------ */
static double dltar4(double wvno, double omga,
                     float *d, float *a, float *b, float *rho,
                     float * /*rtp*/, float * /*dtp*/, float * /*btp*/,
                     int mmax, int llw)
{
    double omega = (omga < 1.0e-4) ? 1.0e-4 : omga;
    double wvno2 = wvno * wvno;

    double xka   = omega / (double)a[mmax - 1];
    double xkb   = omega / (double)b[mmax - 1];
    double wvnop = wvno + xka;
    double wvnom = std::fabs(wvno - xka);
    double ra    = std::sqrt(wvnop * wvnom);
    wvnop = wvno + xkb;
    wvnom = std::fabs(wvno - xkb);
    double rb    = std::sqrt(wvnop * wvnom);
    double t_val = (double)b[mmax - 1] / omega;
    double gammk = 2.0 * t_val * t_val;
    double gam   = gammk * wvno2;
    double gamm1 = gam - 1.0;
    double rho1  = (double)rho[mmax - 1];

    double e[5];
    e[0] = rho1 * rho1 * (gamm1*gamm1 - gam*gammk*ra*rb);
    e[1] = -rho1 * ra;
    e[2] =  rho1 * (gamm1 - gammk*ra*rb);
    e[3] =  rho1 * rb;
    e[4] =  wvno2 - ra*rb;

    for (int m = mmax - 2; m >= llw - 1; m--) {
        xka   = omega / (double)a[m];
        xkb   = omega / (double)b[m];
        t_val = (double)b[m] / omega;
        gammk = 2.0 * t_val * t_val;
        gam   = gammk * wvno2;
        wvnop = wvno + xka;
        wvnom = std::fabs(wvno - xka);
        ra    = std::sqrt(wvnop * wvnom);
        wvnop = wvno + xkb;
        wvnom = std::fabs(wvno - xkb);
        rb    = std::sqrt(wvnop * wvnom);
        double dpth = (double)d[m];
        rho1  = (double)rho[m];
        double p = ra * dpth;
        double q = rb * dpth;

        double w, cosp, exa, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz;
        var_func(p, q, ra, rb, wvno, xka, xkb, dpth,
                 w, cosp, exa, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz);

        double ca[5][5];
        dnka(ca, wvno2, gam, gammk, rho1,
             a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz);

        double ee[5];
        for (int i = 0; i < 5; i++) {
            double cr = 0.0;
            for (int j = 0; j < 5; j++)
                cr += e[j] * ca[j][i];
            ee[i] = cr;
        }
        normc(ee, exa);
        for (int i = 0; i < 5; i++)
            e[i] = ee[i];
    }

    if (llw != 1) {
        /* water layer at surface */
        xka   = omega / (double)a[0];
        wvnop = wvno + xka;
        wvnom = std::fabs(wvno - xka);
        ra    = std::sqrt(wvnop * wvnom);
        double dpth = (double)d[0];
        rho1  = (double)rho[0];
        double p    = ra * dpth;
        double znul = 1.0e-5;
        double w, cosp, exa, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz;
        var_func(p, znul, ra, znul, wvno, xka, znul, dpth,
                 w, cosp, exa, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz);
        double w0 = -rho1 * w;
        return cosp * e[0] + w0 * e[1];
    } else {
        return e[0];
    }
}

/* ------------------------------------------------------------------ */
/* dltar: dispatch to Love (kk=1) or Rayleigh (kk=2) period equation  */
/* ------------------------------------------------------------------ */
static double dltar(double wvno, double omega, int kk,
                    float *d, float *a, float *b, float *rho,
                    float *rtp, float *dtp, float *btp, int mmax, int llw)
{
    if (kk == 1)
        return dltar1(wvno, omega, d, a, b, rho, rtp, dtp, btp, mmax, llw);
    else
        return dltar4(wvno, omega, d, a, b, rho, rtp, dtp, btp, mmax, llw);
}

/* ------------------------------------------------------------------ */
/* half_func: interval bisection step                                   */
/* ------------------------------------------------------------------ */
static void half_func(double c1, double c2, double &c3, double &del3,
                      double omega, int ifunc,
                      float *d, float *a, float *b, float *rho,
                      float *rtp, float *dtp, float *btp, int mmax, int llw)
{
    c3   = 0.5 * (c1 + c2);
    double wvno = omega / c3;
    del3 = dltar(wvno, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw);
}

/* ------------------------------------------------------------------ */
/* nevill: hybrid bisection/Neville root refinement                    */
/* ------------------------------------------------------------------ */
static void nevill(double t, double c1, double c2, double del1, double del2,
                   int ifunc, double &cc,
                   float *d, float *a, float *b, float *rho,
                   float *rtp, float *dtp, float *btp, int mmax, int llw)
{
    double omega = TWOPI / t;
    double c3, del3;
    half_func(c1, c2, c3, del3, omega, ifunc,
              d, a, b, rho, rtp, dtp, btp, mmax, llw);

    int nev   = 1;
    int nctrl = 1;
    int m     = 1;
    double x[20] = {}, y[20] = {};

    while (true) {
        nctrl++;
        if (nctrl >= 100) break;

        if (c3 < std::min(c1, c2) || c3 > std::max(c1, c2)) {
            nev = 0;
            half_func(c1, c2, c3, del3, omega, ifunc,
                      d, a, b, rho, rtp, dtp, btp, mmax, llw);
        }

        double s13 = del1 - del3;
        double s32 = del3 - del2;

        if (std::copysign(1.0, del3) * std::copysign(1.0, del1) < 0.0) {
            c2 = c3;  del2 = del3;
        } else {
            c1 = c3;  del1 = del3;
        }

        if (std::fabs(c1 - c2) <= 1.0e-6 * c1) break;

        if (std::copysign(1.0, s13) != std::copysign(1.0, s32)) nev = 0;

        double ss1 = std::fabs(del1);
        double ss2 = std::fabs(del2);
        double s1  = 0.01 * ss1;
        double s2  = 0.01 * ss2;

        bool do_half = (s1 > ss2 || s2 > ss1 || nev == 0);

        if (do_half) {
            half_func(c1, c2, c3, del3, omega, ifunc,
                      d, a, b, rho, rtp, dtp, btp, mmax, llw);
            nev = 1;
            m   = 1;
        } else {
            /*
             * Neville iteration.
             * C++ 0-based mapping: Fortran x(k) -> x[k-1].
             * x[0..m] hold m+1 points; x[m] is the newest point.
             */
            if (nev == 2) {
                x[m] = c3;
                y[m] = del3;
            } else {
                x[0] = c1;  y[0] = del1;
                x[1] = c2;  y[1] = del2;
                m    = 1;
            }

            bool fallback = false;
            for (int kk = 1; kk <= m; kk++) {
                int j0    = m - kk;           /* Fortran j = m-kk+1 -> 0-based j0 */
                double denom = y[m] - y[j0];  /* Fortran y(m+1) - y(j) */
                if (std::fabs(denom) < 1.0e-10 * std::fabs(y[m])) {
                    fallback = true;
                    break;
                }
                x[j0] = (-y[j0] * x[j0 + 1] + y[m] * x[j0]) / denom;
            }

            if (fallback) {
                half_func(c1, c2, c3, del3, omega, ifunc,
                          d, a, b, rho, rtp, dtp, btp, mmax, llw);
                nev = 1;
                m   = 1;
            } else {
                c3 = x[0];
                double wvno = omega / c3;
                del3 = dltar(wvno, omega, ifunc,
                             d, a, b, rho, rtp, dtp, btp, mmax, llw);
                nev = 2;
                m++;
                if (m > 10) m = 10;
            }
        }
    }

    cc = c3;
}

/* ------------------------------------------------------------------ */
/* getsol: bracket dispersion curve zero, then refine via nevill       */
/* del1st is thread-local state (Fortran save/common equivalent)       */
/* ------------------------------------------------------------------ */
static void getsol(double t1, double &c1, double clow, double dc, double cm,
                   double betmx, int &iret, int ifunc, int ifirst,
                   float *d, float *a, float *b, float *rho,
                   float *rtp, float *dtp, float *btp, int mmax, int llw,
                   double &del1st, GetsolTrace *trace)
{
    double omega = TWOPI / t1;
    double wvno  = omega / c1;
    double del1  = dltar(wvno, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw);

    if (ifirst == 1) del1st = del1;

    double plmn = std::copysign(1.0, del1st) * std::copysign(1.0, del1);
    int idir;
    if (ifirst == 1)
        idir = 1;
    else if (plmn >= 0.0)
        idir = 1;
    else
        idir = -1;

    GetsolTrace local_trace;
    GetsolTrace &tr = trace ? *trace : local_trace;
    tr = GetsolTrace();
    tr.c_start   = c1;
    tr.del_start = del1;
    tr.idir      = idir;

    /* A healthy search needs at most ~(betmx+dc)/dc steps up plus the same
     * down; the cap only stops endless loops on non-finite values (NaN
     * compares false everywhere, so the exit tests below never fire). */
    long max_steps = 1000000;
    if (std::isfinite(betmx) && dc > 0.0)
        max_steps = std::min(max_steps, 10L * (long)((betmx + dc) / dc) + 1000L);

    while (true) {
        if (++tr.nstep > max_steps) {
            tr.c_end  = c1;
            tr.reason = GetsolTrace::STEP_CAP;
            iret = -1;
            return;
        }

        double c2;
        if (idir > 0)
            c2 = c1 + dc;
        else
            c2 = c1 - dc;

        if (c2 <= clow) {
            idir = 1;
            c1   = clow;
            continue;
        }

        wvno = omega / c2;
        double del2 = dltar(wvno, omega, ifunc,
                            d, a, b, rho, rtp, dtp, btp, mmax, llw);

        if (std::copysign(1.0, del1) != std::copysign(1.0, del2)) {
            double cn;
            nevill(t1, c1, c2, del1, del2, ifunc, cn,
                   d, a, b, rho, rtp, dtp, btp, mmax, llw);
            c1   = cn;
            iret = (c1 > betmx) ? -1 : 1;
            tr.c_end  = c1;
            tr.reason = (iret == -1) ? GetsolTrace::ROOT_ABOVE_BETMX : GetsolTrace::OK;
            return;
        }

        c1   = c2;
        del1 = del2;
        if (c1 < cm || c1 >= betmx + dc) {
            iret = -1;
            tr.c_end  = c1;
            tr.reason = (c1 < cm) ? GetsolTrace::BELOW_CM : GetsolTrace::PASSED_BETMX;
            return;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Failure report helpers                                               */
/* ------------------------------------------------------------------ */

/* Scan the period equation on a fine grid and list its sign changes, to
 * tell apart "no root exists below betmx" from "a root exists but the
 * dc-stepping missed it or searched in the wrong direction". */
static std::string scan_period_equation(double t, int ifunc, double cmin, double cmax,
                                        double dc,
                                        float *d, float *a, float *b, float *rho,
                                        float *rtp, float *dtp, float *btp,
                                        int mmax, int llw, double betmx)
{
    std::ostringstream os;
    if (!(std::isfinite(cmin) && std::isfinite(cmax)) || cmax <= cmin || cmin <= 0.0) {
        os << "    scan skipped: invalid range [" << cmin << ", " << cmax << "]\n";
        return os.str();
    }
    const int max_eval = 20000;
    double step = 0.25 * dc;
    if ((cmax - cmin) / step > max_eval) step = (cmax - cmin) / max_eval;

    const double omega = TWOPI / t;
    int n_change = 0, n_below_betmx = 0, n_nonfinite = 0;
    std::ostringstream roots;
    double c_prev = cmin;
    double f_prev = dltar(omega / c_prev, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw);
    if (!std::isfinite(f_prev)) ++n_nonfinite;
    for (double c = cmin + step; c <= cmax + 0.5 * step; c += step) {
        double f = dltar(omega / c, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw);
        if (!std::isfinite(f)) { ++n_nonfinite; c_prev = c; f_prev = f; continue; }
        if (std::isfinite(f_prev) && std::copysign(1.0, f) != std::copysign(1.0, f_prev)) {
            ++n_change;
            if (0.5 * (c + c_prev) <= betmx) ++n_below_betmx;
            if (n_change <= 8) roots << " " << 0.5 * (c + c_prev);
        }
        c_prev = c;
        f_prev = f;
    }
    os << "    scan of period equation, c in [" << cmin << ", " << cmax << "] step " << step
       << ": " << n_change << " sign change(s), " << n_below_betmx << " at c <= betmx";
    if (n_nonfinite > 0) os << ", " << n_nonfinite << " NON-FINITE value(s)";
    os << "\n";
    if (n_change > 0) os << "      sign changes near c =" << roots.str()
                         << (n_change > 8 ? " ..." : "") << "\n";
    return os.str();
}

/* Sanity checks and a dump of the (unflattened) input model. */
static std::string describe_model(const float *thkm, const float *vpm, const float *vsm,
                                  const float *rhom, int nlayer, bool with_table)
{
    std::ostringstream os;
    int n_nonfinite = 0, n_vs_neg = 0, n_vs_zero_below_top = 0, n_vp_nonpos = 0,
        n_rho_nonpos = 0, n_vp_le_vs = 0;
    double vpvs_min = 1.0e30;
    int    vpvs_min_layer = -1;
    int    jmax = -1;
    for (int i = 0; i < nlayer; i++) {
        if (!std::isfinite(thkm[i]) || !std::isfinite(vpm[i]) ||
            !std::isfinite(vsm[i])  || !std::isfinite(rhom[i])) {
            ++n_nonfinite;
            continue;
        }
        if (vsm[i] < 0.0f) ++n_vs_neg;
        if (vsm[i] == 0.0f && i > 0) ++n_vs_zero_below_top;
        if (vpm[i] <= 0.0f) ++n_vp_nonpos;
        if (rhom[i] <= 0.0f) ++n_rho_nonpos;
        if (vsm[i] > 0.0f) {
            if (vpm[i] <= vsm[i]) ++n_vp_le_vs;
            double r = (double)vpm[i] / (double)vsm[i];
            if (r < vpvs_min) { vpvs_min = r; vpvs_min_layer = i; }
        }
        if (jmax < 0 || vsm[i] > vsm[jmax]) jmax = i;
    }
    const int ih = nlayer - 1;
    os << "    model checks: non-finite layers=" << n_nonfinite
       << ", vs<0: " << n_vs_neg
       << ", vs==0 below top: " << n_vs_zero_below_top
       << ", vp<=0: " << n_vp_nonpos
       << ", rho<=0: " << n_rho_nonpos
       << ", vp<=vs: " << n_vp_le_vs;
    if (vpvs_min_layer >= 0)
        os << ", min vp/vs=" << vpvs_min << " (layer " << vpvs_min_layer << ")";
    os << "\n";
    os << "    halfspace (layer " << ih << ") vs=" << vsm[ih];
    if (jmax >= 0) {
        os << "; max vs=" << vsm[jmax] << " at layer " << jmax;
        if (vsm[ih] < vsm[jmax])
            os << "  <-- halfspace is SLOWER than an overlying layer "
                  "(fundamental mode can exceed halfspace Vs at long periods)";
    }
    os << "\n";

    if (with_table) {
        os << "    input model (before earth flattening):\n"
           << "      layer   top_km    thk_km      vp      vs     rho\n";
        double z = 0.0;
        char line[160];
        for (int i = 0; i < nlayer; i++) {
            snprintf(line, sizeof(line), "      %5d %8.3f %9.4f %7.4f %7.4f %7.4f\n",
                     i, z, (double)thkm[i], (double)vpm[i], (double)vsm[i], (double)rhom[i]);
            os << line;
            z += (double)thkm[i];
        }
    }
    return os.str();
}

/* ================================================================== */
/* surfdisp96_cpp: low-level implementation (Fortran-style interface)  */
/* Prefer surfdisp::dispersion() for new C++ code.                    */
/* ================================================================== */
std::vector<double> disper(const float *thkm, const float *vpm, const float *vsm,
                           const float *rhom, int nlayer, int iflsph, int iwave,
                           int mode, int igr, int kmax,
                           const double *t)
{
    int mmax = nlayer;
    int nsph = iflsph;
    std::vector<double> cg(kmax, 0.0);

    std::vector<float> d(mmax), a(mmax), b(mmax), rho(mmax);
    std::vector<float> rtp(mmax), dtp(mmax), btp(mmax);

    for (int i = 0; i < mmax; i++) {
        b[i]   = vsm[i];
        a[i]   = vpm[i];
        d[i]   = thkm[i];
        rho[i] = rhom[i];
    }

    int idispl = 0, idispr = 0;
    if (iwave == 1) idispl = kmax;
    else            idispr = kmax;

    const char *wave_name = (iwave == 1) ? "Love" : "Rayleigh";

    /* Non-finite values, negative/zero Vs below the top layer, or
     * non-positive Vp/rho make the root search fail or loop forever.
     * Report them explicitly instead. */
    {
        bool bad = false;
        for (int i = 0; i < mmax && !bad; i++) {
            bad = !std::isfinite(thkm[i]) || !std::isfinite(vpm[i]) ||
                  !std::isfinite(vsm[i])  || !std::isfinite(rhom[i]) ||
                  vsm[i] < 0.0f || (vsm[i] == 0.0f && i > 0) ||
                  vpm[i] <= 0.0f || rhom[i] <= 0.0f || vpm[i] <= vsm[i];
        }
        if (bad) {
            std::ostringstream os;
            os << "WARNING: improper initial value in disper - no zero found in fundamental mode\n"
               << "  disper: INVALID INPUT MODEL (" << wave_name << ", igr=" << igr
               << ", nlayer=" << nlayer << ", kmax=" << kmax << "); returning zeros for all periods\n"
               << describe_model(thkm, vpm, vsm, rhom, nlayer, true);
            record_failure(std::string("WARNING: disper: invalid input model (") + wave_name +
                           "), returning zeros", os.str());
            return cg;
        }
    }

    /* Phase velocity search increment */
    float min_vs = *std::min_element(vsm, vsm + nlayer);
    float ddc0;
    if      (min_vs < 1.0f) ddc0 = 0.001f;
    else if (min_vs < 2.0f) ddc0 = 0.002f;
    else                    ddc0 = 0.005f;

    float sone0 = 1.5f;
    double h0   = 0.005;

    int llw = (b[0] <= 0.0f) ? 2 : 1;

    float dhalf = 0.0f;
    if (nsph == 1)
        sphere_func(0, 0, d.data(), a.data(), b.data(), rho.data(),
                    rtp.data(), dtp.data(), btp.data(), mmax, llw, dhalf);

    /* Find extremal velocities */
    float betmx = -1.0e20f, betmn = 1.0e20f;
    int   jmn = 0, jsol = 1;
    for (int i = 0; i < mmax; i++) {
        if (b[i] > 0.01f && b[i] < betmn) {
            betmn = b[i];  jmn = i;  jsol = 1;
        } else if (b[i] <= 0.01f && a[i] < betmn) {
            betmn = a[i];  jmn = i;  jsol = 0;
        }
        if (b[i] > betmx) betmx = b[i];
    }

    std::vector<double> c(kmax, 0.0), cb(kmax, 0.0);
    double one = 1.0e-2;
    int    iverb[2] = {0, 0};

    for (int ifunc = 1; ifunc <= 2; ifunc++) {
        if (ifunc == 1 && idispl <= 0) continue;
        if (ifunc == 2 && idispr <= 0) continue;

        if (nsph == 1)
            sphere_func(ifunc, 1, d.data(), a.data(), b.data(), rho.data(),
                        rtp.data(), dtp.data(), btp.data(), mmax, llw, dhalf);

        float  ddc  = ddc0;
        float  sone = sone0;
        double h    = h0;

        if (sone < 0.01f) sone = 2.0f;
        double onea = (double)sone;

        float cc1;
        if (jsol == 0)
            cc1 = betmn;
        else
            gtsolh(a[jmn], b[jmn], cc1);

        cc1 *= 0.95f;
        cc1 *= 0.90f;

        double cc = (double)cc1;
        double dc = std::fabs((double)ddc);
        double c1 = cc;
        double cm_val = cc;

        for (int i = 0; i < kmax; i++) { c[i] = 0.0;  cb[i] = 0.0; }

        int ift = 999;
        double del1st = 0.0;   /* saved state for getsol direction logic */

        GetsolTrace trace;

        /* Build and record a report for a fundamental-mode failure at period k. */
        auto report_failure = [&](int k, double t_search, const char *what) {
            std::ostringstream os;
            os << "WARNING: improper initial value in disper - no zero found in fundamental mode\n"
               << "  disper failure: " << wave_name << (igr > 0 ? " group" : " phase")
               << " (igr=" << igr << "), mode " << mode
               << ", " << (nsph == 1 ? "spherical" : "flat") << " earth"
               << ", nlayer=" << nlayer << ", kmax=" << kmax
               << (llw != 1 ? ", water layer on top" : "") << "\n"
               << "  failed at period index k=" << k << " of " << kmax
               << ": T=" << t[k] << " s (searched at T=" << t_search << " s)\n";
            if (k > 0) {
                os << "  phase velocities at earlier periods (km/s):";
                for (int i = 0; i < k; i++) os << " " << c[i];
                os << "\n";
            } else {
                os << "  failed at the FIRST period: the search from the starting value found no root\n";
            }
            os << "  " << what << "\n"
               << "  getsol: " << getsol_reason_str(trace.reason)
               << "; start c=" << trace.c_start
               << (k == 0 ? " (= cm)" : " (= previous c - 1.5*dc)")
               << ", period equation at start=" << trace.del_start
               << ", initial direction=" << (trace.idir > 0 ? "up" : "down")
               << ", steps=" << trace.nstep
               << ", last c=" << trace.c_end << "\n"
               << "  search parameters (flattened model): cm=" << cm_val
               << " (0.855 x " << (jsol == 0 ? "Vp" : "halfspace Rayleigh velocity")
               << " of slowest layer " << jmn << "), dc=" << dc
               << ", betmn=" << betmn << ", betmx=" << betmx
               << ", flattened halfspace vs=" << b[mmax - 1] << "\n"
               << scan_period_equation(t_search, ifunc, 0.9 * cm_val, 1.1 * (double)betmx, dc,
                                       d.data(), a.data(), b.data(), rho.data(),
                                       rtp.data(), dtp.data(), btp.data(),
                                       mmax, llw, (double)betmx)
               << describe_model(thkm, vpm, vsm, rhom, nlayer, true);

            std::ostringstream summary;
            summary << "WARNING: disper: no fundamental-mode zero (" << wave_name
                    << ", igr=" << igr << ") at T=" << t[k] << " s (k=" << k << "/" << kmax
                    << "): " << getsol_reason_str(trace.reason)
                    << "; betmx=" << betmx << ", last c=" << trace.c_end;
            record_failure(summary.str(), os.str());
        };

        if (!std::isfinite(cc) || cc <= 0.0) {
            trace = GetsolTrace();
            trace.c_start = trace.c_end = cc;
            report_failure(0, t[0], "non-finite or non-positive starting velocity from gtsolh "
                                    "(check vp/vs of the slowest layer)");
            for (int i = 0; i < kmax; i++) cg[i] = 0.0;
            return cg;
        }

        for (int iq = 1; iq <= mode; iq++) {
            int is = 0;
            int ie = kmax - 1;

            bool mode_failed = false;
            int  fail_k      = 0;

            for (int k = is; k <= ie; k++) {

                if (k >= ift) {
                    /* Higher mode failed at an earlier period */
                    if (iq > 1) {
                        mode_failed = true;
                        fail_k = k;
                    } else {
                        if (iverb[ifunc - 1] == 0) {
                            iverb[ifunc - 1] = 1;
                            trace = GetsolTrace();
                            report_failure(k, t[k], "fundamental mode marked as failed at an earlier period");
                        }
                        for (int i = k; i <= ie; i++) cg[i] = 0.0;
                        return cg;
                    }
                    break;
                }

                /* Period to search at */
                double t1 = t[k];
                double t1a, t1b;
                if (igr > 0) {
                    t1a = t1 / (1.0 + h);
                    t1b = t1 / (1.0 - h);
                    t1  = t1a;
                } else {
                    t1a = t[k];
                    t1b = 0.0;
                }

                /* Initial phase velocity estimate for bracketing */
                double clow;
                int    ifirst;
                if (k == is && iq == 1) {
                    c1     = cc;
                    clow   = cc;
                    ifirst = 1;
                } else if (k == is && iq > 1) {
                    c1     = c[is] + one * dc;
                    clow   = c1;
                    ifirst = 1;
                } else if (k > is && iq > 1) {
                    ifirst = 0;
                    clow   = c[k] + one * dc;
                    c1     = c[k - 1];
                    if (c1 < clow) c1 = clow;
                } else {            /* k > is && iq == 1 */
                    ifirst = 0;
                    c1     = c[k - 1] - onea * dc;
                    clow   = cm_val;
                }

                int iret;
                getsol(t1, c1, clow, dc, cm_val, (double)betmx,
                       iret, ifunc, ifirst,
                       d.data(), a.data(), b.data(), rho.data(),
                       rtp.data(), dtp.data(), btp.data(),
                       mmax, llw, del1st, &trace);

                if (iret == -1) {
                    if (iq > 1) {
                        mode_failed = true;
                        fail_k = k;
                    } else {
                        if (iverb[ifunc - 1] == 0) {
                            iverb[ifunc - 1] = 1;
                            report_failure(k, t1, "root search (getsol) failed");
                        }
                        for (int i = k; i <= ie; i++) cg[i] = 0.0;
                        return cg;
                    }
                    break;
                }

                c[k] = c1;

                /* Group velocity: solve at t1b and finite-difference */
                if (igr > 0) {
                    ifirst = 0;
                    clow   = cb[k] + one * dc;
                    c1     = c1 - onea * dc;
                    getsol(t1b, c1, clow, dc, cm_val, (double)betmx,
                           iret, ifunc, ifirst,
                           d.data(), a.data(), b.data(), rho.data(),
                           rtp.data(), dtp.data(), btp.data(),
                           mmax, llw, del1st);
                    if (iret == -1) c1 = c[k];
                    cb[k] = c1;
                } else {
                    c1 = 0.0;
                }

                /* Intentional precision reduction to match Fortran sngl() */
                float cc0 = (float)c[k];
                float cc1_v = (float)c1;

                if (igr == 0) {
                    cg[k] = (double)cc0;
                } else {
                    double gvel = (1.0 / t1a - 1.0 / t1b)
                                / (1.0 / (t1a * cc0) - 1.0 / (t1b * cc1_v));
                    cg[k] = gvel;
                }
            } /* end k loop */

            if (mode_failed) {
                ift = fail_k;
                for (int i = fail_k; i <= ie; i++) cg[i] = 0.0;
            }
        } /* end iq loop */
    } /* end ifunc loop */
    return cg;
}
