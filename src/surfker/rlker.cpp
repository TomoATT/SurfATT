/*
 * C++ port of sregn96.f90 (Rayleigh-wave eigenfunctions)
 * Original: C. Y. Wang, R. B. Herrmann (1981), D. R. Russell (1984),
 *           R. B. Herrmann (2010), Nanqiao Du (2021), Shijie Hao (2024)
 *
 * Array indexing: Fortran 1-based -> C++ 0-based throughout.
 *
 * Design:
 *   - thread_local RayleighWork: zero heap allocation per call
 *   - All internal functions are static
 *   - std::complex<double> (cd) used where Fortran uses complex(kind=dp)
 */

#include "surfdisp.hpp"
#include <cmath>
#include <cstring>
#include <complex>
#include <vector>
#include <stdexcept>


using cd = std::complex<double>;

// ============================================================
// Fixed-size workspace (zero heap allocation after first call)
// ============================================================

struct RayleighWork {
    int  mmax;
    bool allfluid;
    double zd[MAX_LAYERS], za[MAX_LAYERS], zb[MAX_LAYERS];
    double zrho[MAX_LAYERS], xmu[MAX_LAYERS], xlam[MAX_LAYERS];
    int    iwat[MAX_LAYERS];
    double ur[MAX_LAYERS], uz[MAX_LAYERS], tz[MAX_LAYERS], tr[MAX_LAYERS];
    double dcda[MAX_LAYERS], dcdb[MAX_LAYERS], dcdr[MAX_LAYERS];
    double dcdh[MAX_LAYERS], dcdgc[MAX_LAYERS], dcdgs[MAX_LAYERS];
    double sumi0, sumi1, sumi2, sumi3, flagr, are, ugr;
    double vtp[MAX_LAYERS], dtp[MAX_LAYERS], rtp[MAX_LAYERS];
    double exe[MAX_LAYERS];   // scaling for Dunkin compound vectors
    double exa[MAX_LAYERS];   // scaling for Haskell vectors
    double cd_arr[MAX_LAYERS][5];  // Dunkin compound matrix rows cd(m,1..5)
    double vv[MAX_LAYERS][4];      // Haskell propagator column vv(m,1..4)
    cd     e[4][4], einv[4][4];    // E and E^{-1} matrices (complex 4x4)
    cd     ra, rb;                 // complex vertical wavenumbers for halfspace
    double uu0[4];
    // sregnpu temporaries (on workspace, no heap allocation)
    double dcda1[MAX_LAYERS], dcdb1[MAX_LAYERS], dcdr1[MAX_LAYERS], dcdh1[MAX_LAYERS];
    double dcda2[MAX_LAYERS], dcdb2[MAX_LAYERS], dcdr2[MAX_LAYERS], dcdh2[MAX_LAYERS];
    double dcda_t[MAX_LAYERS], dcdb_t[MAX_LAYERS], dcdr_t[MAX_LAYERS], dcdh_t[MAX_LAYERS];
    double ur_t[MAX_LAYERS], uz_t[MAX_LAYERS], tz_t[MAX_LAYERS], tr_t[MAX_LAYERS];
};

static thread_local RayleighWork tl_work;

// ============================================================
// normc: normalize vector, store log(max) in ex
// ============================================================
static void normc(double* ee, double& ex, int nmat)
{
    double t1 = 0.0;
    for (int i = 0; i < nmat; i++) {
        double a = std::fabs(ee[i]);
        if (a > t1) t1 = a;
    }
    if (t1 < 1.0e-40) t1 = 1.0;
    for (int i = 0; i < nmat; i++)
        ee[i] /= t1;
    ex = std::log(t1);
}

// ============================================================
// bldsph: spherical-to-flat earth transform (Rayleigh waves)
// ============================================================
static void bldsph(RayleighWork& s)
{
    const double ar = 6371.0;
    double dr = 0.0;
    double r0 = ar;

    for (int i = 0; i < s.mmax; i++) {
        double dz = (i == s.mmax - 1) ? 1.0 : s.zd[i];
        dr += dz;
        double r1 = ar - dr;
        double z0 = ar * std::log(ar / r0);
        double z1 = ar * std::log(ar / r1);

        double tmp = (2.0 * ar) / (r0 + r1);
        s.vtp[i]    = tmp;
        s.rtp[i]    = std::pow(tmp, -2.275);  // Rayleigh: -2.275 (not -5)
        s.dtp[i]    = ar / r0;

        s.za[i]   *= tmp;
        s.zb[i]   *= tmp;
        s.zrho[i] *= s.rtp[i];
        s.zd[i]    = z1 - z0;
        r0 = r1;
    }
    s.zd[s.mmax - 1] = 0.0;
}

// ============================================================
// varsv: per-layer propagator trigonometric/hyperbolic functions
// ============================================================
static void varsv(cd p, cd q, cd rp, cd rsv,
                  double& cosp, double& cosq,
                  double& rsinp, double& rsinq,
                  double& sinpr, double& sinqr,
                  double& pex, double& svex,
                  int iwat, double zd_m)
{
    pex  = p.real();
    svex = 0.0;

    // P wave part
    {
        double pi_val = p.imag();
        cd epp = cd(std::cos(pi_val), std::sin(pi_val)) / 2.0;
        cd epm = std::conj(epp);
        double pfac = (p.real() < 30.0) ? std::exp(-2.0 * p.real()) : 0.0;
        cosp = std::real(epp + pfac * epm);
        cd sinp = epp - pfac * epm;
        rsinp = std::real(rp * sinp);
        if (std::fabs(p.real()) < 1.0e-5 && std::abs(rp) < 1.0e-5)
            sinpr = zd_m;
        else
            sinpr = std::real(sinp / rp);
    }

    if (iwat == 1) {
        // fluid layer: no SV part
        cosq  = 1.0;
        rsinq = 0.0;
        sinqr = 0.0;
    } else {
        // elastic layer: SV part
        svex = q.real();
        double qi_val = q.imag();
        cd eqp = cd(std::cos(qi_val), std::sin(qi_val)) / 2.0;
        cd eqm = std::conj(eqp);
        double svfac = (q.real() < 30.0) ? std::exp(-2.0 * q.real()) : 0.0;
        cosq = std::real(eqp + svfac * eqm);
        cd sinq = eqp - svfac * eqm;
        rsinq = std::real(rsv * sinq);
        if (std::fabs(q.real()) < 1.0e-5 && std::abs(rsv) < 1.0e-5)
            sinqr = zd_m;
        else
            sinqr = std::real(sinq / rsv);
    }
}

// ============================================================
// evalg: set up halfspace E/EINV matrices and boundary vector gbr
// m0 is 0-based layer index
// ============================================================
static void evalg(int jbdry, int m0, cd gbr[][5], int inp,
                  double wvno, double om, double om2, double wvno2,
                  RayleighWork& s)
{
    double xka = om / s.za[m0];
    double xkb = (s.zb[m0] > 0.0) ? om / s.zb[m0] : 0.0;

    s.ra = std::sqrt(cd(wvno2 - xka * xka, 0.0));
    s.rb = std::sqrt(cd(wvno2 - xkb * xkb, 0.0));

    double gam   = s.zb[m0] * wvno / om;
    gam = 2.0 * gam * gam;
    double gamm1 = std::real(gam - cd(1.0, 0.0));

    double rho = s.zrho[m0];

    if (jbdry < 0) {
        // RIGID
        if (s.zb[m0] > 0.0) {
            // elastic above - rigid
            gbr[inp][0] = cd(1, 0);
            gbr[inp][1] = cd(0, 0);
            gbr[inp][2] = cd(0, 0);
            gbr[inp][3] = cd(0, 0);
            gbr[inp][4] = cd(0, 0);
        } else {
            // fluid above - rigid
            gbr[inp][0] = cd(0, 0);
            gbr[inp][1] = cd(0, 0);
            gbr[inp][2] = cd(0, 0);
            gbr[inp][3] = cd(0, 0);
            gbr[inp][4] = cd(0, 0);
            if (s.allfluid)
                gbr[inp][0] = cd(1, 0);
            else
                gbr[inp][3] = cd(1, 0);
        }
    } else if (jbdry == 0) {
        // HALFSPACE
        if (s.iwat[m0] == 0) {
            // elastic halfspace
            s.e[0][0] = wvno;     s.e[0][1] = s.rb;    s.e[0][2] = wvno;    s.e[0][3] = -s.rb;
            s.e[1][0] = s.ra;     s.e[1][1] = wvno;    s.e[1][2] = -s.ra;   s.e[1][3] = wvno;
            s.e[2][0] = rho*om2*gamm1;
            s.e[2][1] = rho*om2*gam*s.rb/wvno;
            s.e[2][2] = rho*om2*gamm1;
            s.e[2][3] = -rho*om2*gam*s.rb/wvno;
            s.e[3][0] = rho*om2*gam*s.ra/wvno;
            s.e[3][1] = rho*om2*gamm1;
            s.e[3][2] = -rho*om2*gam*s.ra/wvno;
            s.e[3][3] = rho*om2*gamm1;

            s.einv[0][0] =  0.5*gam/wvno;
            s.einv[0][1] = -0.5*gamm1/s.ra;
            s.einv[0][2] = -0.5/(rho*om2);
            s.einv[0][3] =  0.5*wvno/(rho*om2*s.ra);

            s.einv[1][0] = -0.5*gamm1/s.rb;
            s.einv[1][1] =  0.5*gam/wvno;
            s.einv[1][2] =  0.5*wvno/(rho*om2*s.rb);
            s.einv[1][3] = -0.5/(rho*om2);

            s.einv[2][0] =  0.5*gam/wvno;
            s.einv[2][1] =  0.5*gamm1/s.ra;
            s.einv[2][2] = -0.5/(rho*om2);
            s.einv[2][3] = -0.5*wvno/(rho*om2*s.ra);

            s.einv[3][0] =  0.5*gamm1/s.rb;
            s.einv[3][1] =  0.5*gam/wvno;
            s.einv[3][2] = -0.5*wvno/(rho*om2*s.rb);
            s.einv[3][3] = -0.5/(rho*om2);

            // gbr: compound vector (normalized)
            cd norm = -cd(rho*rho)*om2*om2*wvno2*s.ra*s.rb;
            gbr[inp][0] = 0.25*(cd(rho*rho)*om2*om2*(-gam*gam*s.ra*s.rb + wvno2*gamm1*gamm1)) / norm;
            gbr[inp][1] = 0.25*(-cd(rho)*wvno2*s.ra*om2) / norm;
            gbr[inp][2] = 0.25*(-cd(rho)*(-gam*s.ra*s.rb + wvno2*gamm1)*om2*wvno) / norm;
            gbr[inp][3] = 0.25*(cd(rho)*wvno2*s.rb*om2) / norm;
            gbr[inp][4] = 0.25*(wvno2*(wvno2 - s.ra*s.rb)) / norm;

        } else {
            // fluid halfspace
            // null E/EINV first
            for (int ii = 0; ii < 4; ii++)
                for (int jj = 0; jj < 4; jj++) {
                    s.e[ii][jj]    = cd(0, 0);
                    s.einv[ii][jj] = cd(0, 0);
                }
            s.e[0][0] =  s.ra;
            s.e[0][1] = -s.ra;
            s.e[1][0] = -rho * om2;
            s.e[1][1] = -rho * om2;

            s.einv[0][0] =  0.5 / s.ra;
            s.einv[0][1] = -0.5 / (rho * om2);
            s.einv[1][0] = -0.5 / s.ra;
            s.einv[1][1] = -0.5 / (rho * om2);

            if (s.allfluid) {
                gbr[inp][0] = 0.5 / s.ra;
                gbr[inp][1] = cd(0.5, 0) / (-rho * om2);
                gbr[inp][2] = cd(0, 0);
                gbr[inp][3] = cd(0, 0);
                gbr[inp][4] = cd(0, 0);
            } else {
                gbr[inp][0] = cd(0, 0);
                gbr[inp][1] = cd(0, 0);
                gbr[inp][2] = cd(0, 0);
                gbr[inp][3] = 0.5 * rho * om2 / s.ra;
                gbr[inp][4] = cd(-0.5, 0);
            }
        }
    } else {
        // FREE surface (jbdry == 1)
        if (s.zb[m0] > 0.0) {
            gbr[inp][0] = cd(0, 0);
            gbr[inp][1] = cd(0, 0);
            gbr[inp][2] = cd(0, 0);
            gbr[inp][3] = cd(0, 0);
            gbr[inp][4] = cd(1, 0);
        } else {
            gbr[inp][0] = cd(0, 0);
            gbr[inp][1] = cd(0, 0);
            gbr[inp][2] = cd(0, 0);
            gbr[inp][3] = cd(0, 0);
            gbr[inp][4] = cd(0, 0);
            if (s.allfluid)
                gbr[inp][1] = cd(1, 0);
            else
                gbr[inp][4] = cd(1, 0);
        }
    }
}

// ============================================================
// dnka: 5x5 Dunkin compound matrix
// ============================================================
static void dnka(double ca[5][5],
                 double cosp, double rsinp, double sinpr,
                 double cossv, double rsinsv, double sinsvr,
                 double rho, double b,
                 int iwat, double ex, double exa_val,
                 double wvno, double wvno2, double om2)
{
    if (iwat == 1) {
        // fluid layer
        for (int j = 0; j < 5; j++)
            for (int i = 0; i < 5; i++)
                ca[i][j] = 0.0;

        double dfac = (ex > 35.0) ? 0.0 : std::exp(-ex);
        ca[2][2] = dfac;
        ca[0][0] = cosp;
        ca[4][4] = cosp;
        ca[0][1] = -rsinp / (rho * om2);
        ca[1][0] = -rho * sinpr * om2;
        ca[1][1] = cosp;
        ca[3][3] = cosp;
        ca[3][4] = ca[0][1];
        ca[4][3] = ca[1][0];
    } else {
        // elastic layer
        double a0 = (exa_val < 60.0) ? std::exp(-exa_val) : 0.0;
        double rho2 = rho * rho;
        double gam   = 2.0 * b * b * wvno2 / om2;
        double gam2  = gam * gam;
        double gamm1 = gam - 1.0;
        double gamm2 = gamm1 * gamm1;

        double cpcq  = cosp * cossv;
        double cpy   = cosp * sinsvr;
        double cpz   = cosp * rsinsv;
        double cqw   = cossv * sinpr;
        double cqx   = cossv * rsinp;
        double xy    = rsinp * sinsvr;
        double xz    = rsinp * rsinsv;
        double wy    = sinpr * sinsvr;
        double wz    = sinpr * rsinsv;

        double cqww2  = cqw * wvno2;
        double cqxw2  = cqx / wvno2;
        double gg1    = gam * gamm1;
        double a0c    = std::real(cd(2.0, 0.0) * (cd(a0, 0.0) - cpcq));
        double xz2    = xz / wvno2;
        double gxz2   = gam * xz2;
        double g2xz2  = gam2 * xz2;
        double a0cgg1 = a0c * (gam + gamm1);
        double wy2    = wy * wvno2;
        double g2wy2  = gamm2 * wy2;
        double g1wy2  = gamm1 * wy2;

        double temp;

        temp = a0c * gg1 + g2xz2 + g2wy2;
        ca[2][2] = a0 + temp + temp;
        ca[0][0] = cpcq - temp;

        ca[0][1] = (-cqx + wvno2 * cpy) / (rho * om2);

        temp = std::real(cd(0.5, 0.0) * a0cgg1 + gxz2 + g1wy2);
        ca[0][2] = wvno * temp / (rho * om2);

        ca[0][3] = (-cqww2 + cpz) / (rho * om2);

        temp = wvno2 * (a0c + wy2) + xz;
        ca[0][4] = -temp / (rho2 * om2 * om2);

        ca[1][0] = (-gamm2 * cqw + gam2 * cpz / wvno2) * rho * om2;
        ca[1][1] = cpcq;
        ca[1][2] = (gamm1 * cqww2 - gam * cpz) / wvno;
        ca[1][3] = -wz;
        ca[1][4] = ca[0][3];

        temp = std::real(cd(0.5, 0.0) * a0cgg1 * gg1 + gam2 * gxz2 + gamm2 * g1wy2);
        ca[2][0] = std::real(cd(-2.0, 0.0) * temp * rho * om2 / wvno);
        ca[2][1] = std::real(-wvno * (gam * cqxw2 - gamm1 * cpy) * cd(2.0, 0.0));
        ca[2][3] = -2.0 * ca[1][2];
        ca[2][4] = -2.0 * ca[0][2];

        ca[3][0] = (-gam2 * cqxw2 + gamm2 * cpy) * rho * om2;
        ca[3][1] = -xy;
        ca[3][2] = -ca[2][1] / 2.0;
        ca[3][3] = ca[1][1];
        ca[3][4] = ca[0][1];

        temp = gamm2 * (a0c * gam2 + g2wy2) + gam2 * g2xz2;
        ca[4][0] = -rho2 * om2 * om2 * temp / wvno2;
        ca[4][1] = ca[3][0];
        ca[4][2] = -ca[2][0] / 2.0;
        ca[4][3] = ca[1][0];
        ca[4][4] = ca[0][0];
    }
}

// ============================================================
// hska: 4x4 Haskell propagator matrix
// ============================================================
static void hska(double AA[4][4],
                 double cosp, double rsinp, double sinpr,
                 double tcossv, double trsinsv, double tsinsvr,
                 double rho, double b,
                 int iwat, double pex, double svex,
                 double wvno, double wvno2, double om2)
{
    if (iwat == 1) {
        // fluid layer
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                AA[i][j] = 0.0;

        double dfac = (pex > 35.0) ? 0.0 : std::exp(-pex);
        AA[0][0] = dfac;
        AA[3][3] = dfac;
        AA[1][1] = cosp;
        AA[2][2] = cosp;
        AA[1][2] = -rsinp / (rho * om2);
        AA[2][1] = -rho * om2 * sinpr;
    } else {
        // elastic layer
        // adjust SV functions for exp(pex) factor (see Fortran comments)
        double dfac = ((pex - svex) > 70.0) ? 0.0 : std::exp(svex - pex);
        double cossv  = dfac * tcossv;
        double rsinsv = dfac * trsinsv;
        double sinsvr = dfac * tsinsvr;

        double gam   = 2.0 * b * b * wvno2 / om2;
        double gamm1 = gam - 1.0;

        AA[0][0] =  cossv + gam * (cosp - cossv);
        AA[0][1] = -wvno * gamm1 * sinpr + gam * rsinsv / wvno;
        AA[0][2] = -wvno * (cosp - cossv) / (rho * om2);
        AA[0][3] =  (wvno2 * sinpr - rsinsv) / (rho * om2);

        AA[1][0] =  gam * rsinp / wvno - wvno * gamm1 * sinsvr;
        AA[1][1] =  cosp - gam * (cosp - cossv);
        AA[1][2] =  (-rsinp + wvno2 * sinsvr) / (rho * om2);
        AA[1][3] = -AA[0][2];

        AA[2][0] =  rho * om2 * gam * gamm1 * (cosp - cossv) / wvno;
        AA[2][1] =  rho * om2 * (-gamm1 * gamm1 * sinpr + gam * gam * rsinsv / wvno2);
        AA[2][2] =  AA[1][1];
        AA[2][3] = -AA[0][1];

        AA[3][0] =  rho * om2 * (gam * gam * rsinp / wvno2 - gamm1 * gamm1 * sinsvr);
        AA[3][1] = -AA[2][0];
        AA[3][2] = -AA[1][0];
        AA[3][3] =  AA[0][0];
    }
}

// ============================================================
// up: propagate Dunkin compound vectors from bottom upward
// ============================================================
static void up_func(double omega, double wvno, double& fr, RayleighWork& s)
{
    int mmax = s.mmax;
    double wvno2 = wvno * wvno;
    double om2   = omega * omega;

    // Initialize base vector at bottom halfspace
    cd gbr[2][5];
    evalg(0, mmax - 1, gbr, 0, wvno, omega, om2, wvno2, s);

    s.cd_arr[mmax - 1][0] = std::real(gbr[0][0]);
    s.cd_arr[mmax - 1][1] = std::real(gbr[0][1]);
    s.cd_arr[mmax - 1][2] = std::real(gbr[0][2]);
    s.cd_arr[mmax - 1][3] = std::real(gbr[0][3]);
    s.cd_arr[mmax - 1][4] = std::real(gbr[0][4]);
    s.exe[mmax - 1] = 0.0;

    double exsum = 0.0;

    for (int m = mmax - 2; m >= 0; m--) {
        double xka = omega / s.za[m];
        double xkb = (s.zb[m] > 0.0) ? omega / s.zb[m] : 0.0;

        cd rp  = std::sqrt(cd(wvno2 - xka * xka, 0.0));
        cd rsv = std::sqrt(cd(wvno2 - xkb * xkb, 0.0));
        cd p   = rp  * s.zd[m];
        cd q   = rsv * s.zd[m];

        double cosp, cosq, rsinp, rsinq, sinpr, sinqr, pex, svex;
        varsv(p, q, rp, rsv, cosp, cosq, rsinp, rsinq, sinpr, sinqr,
              pex, svex, s.iwat[m], s.zd[m]);

        double ca[5][5];
        dnka(ca, cosp, rsinp, sinpr, cosq, rsinq, sinqr,
             s.zrho[m], s.zb[m], s.iwat[m], pex, pex + svex,
             wvno, wvno2, om2);

        // Multiply: ee = cd_arr[m+1] * ca  (row vector times matrix)
        double ee[5];
        for (int i = 0; i < 5; i++) {
            double cr = 0.0;
            for (int j = 0; j < 5; j++)
                cr += s.cd_arr[m + 1][j] * ca[j][i];
            ee[i] = cr;
        }

        double exn = 0.0;
        normc(ee, exn, 5);
        exsum += pex + svex + exn;
        s.exe[m] = exsum;

        for (int i = 0; i < 5; i++)
            s.cd_arr[m][i] = ee[i];
    }

    fr = s.cd_arr[0][0];
}

// ============================================================
// down: propagate Haskell vectors from top downward
// ============================================================
static void down_func(double omega, double wvno, RayleighWork& s)
{
    int mmax = s.mmax;
    double wvno2 = wvno * wvno;
    double om2   = omega * omega;

    // Initialize top surface: first column of Haskell propagator
    s.vv[0][0] = 1.0;
    s.vv[0][1] = 0.0;
    s.vv[0][2] = 0.0;
    s.vv[0][3] = 0.0;
    s.exa[0] = 0.0;

    double exsum = 0.0;

    for (int m = 0; m < mmax - 1; m++) {
        double xka = omega / s.za[m];
        double xkb = (s.zb[m] > 0.0) ? omega / s.zb[m] : 0.0;

        cd rp  = std::sqrt(cd(wvno2 - xka * xka, 0.0));
        cd rsv = std::sqrt(cd(wvno2 - xkb * xkb, 0.0));
        cd p   = rp  * s.zd[m];
        cd q   = rsv * s.zd[m];

        double cosp, cosq, rsinp, rsinq, sinpr, sinqr, pex, svex;
        varsv(p, q, rp, rsv, cosp, cosq, rsinp, rsinq, sinpr, sinqr,
              pex, svex, s.iwat[m], s.zd[m]);

        double AA[4][4];
        hska(AA, cosp, rsinp, sinpr, cosq, rsinq, sinqr,
             s.zrho[m], s.zb[m], s.iwat[m], pex, svex,
             wvno, wvno2, om2);

        // Multiply: aa0 = AA * vv[m]
        double aa0[4];
        for (int i = 0; i < 4; i++) {
            double cc = 0.0;
            for (int j = 0; j < 4; j++)
                cc += AA[i][j] * s.vv[m][j];
            aa0[i] = cc;
        }

        double ex2 = 0.0;
        normc(aa0, ex2, 4);
        exsum += pex + ex2;
        s.exa[m + 1] = exsum;

        for (int i = 0; i < 4; i++)
            s.vv[m + 1][i] = aa0[i];
    }
}

// ============================================================
// ffunc: (1 - exp(-2*nub*dm)) / (2*nub)
// ============================================================
static cd ffunc(cd nub, double dm)
{
    if (std::abs(nub) < 1.0e-8)
        return cd(dm, 0.0);
    cd arg = nub * dm;
    cd exqq = (arg.real() < 40.0) ? std::exp(-2.0 * arg) : cd(0, 0);
    return (1.0 - exqq) / (2.0 * nub);
}

// ============================================================
// gfunc: exp(-nub*dm) * dm
// ============================================================
static cd gfunc(cd nub, double dm)
{
    cd arg = nub * dm;
    return (arg.real() < 75.0) ? std::exp(-arg) * dm : cd(0, 0);
}

// ============================================================
// h1func: (1 - exp(-(nua+nub)*dm)) / (nua+nub)
// ============================================================
static cd h1func(cd nua, cd nub, double dm)
{
    if (std::abs(nua + nub) < 1.0e-8)
        return cd(dm, 0.0);
    cd arg = (nua + nub) * dm;
    cd exqq = (arg.real() < 40.0) ? std::exp(-arg) : cd(0, 0);
    return (1.0 - exqq) / (nua + nub);
}

// ============================================================
// h2func: (exp(-nub*dm) - exp(-nua*dm)) / (nua - nub)
// ============================================================
static cd h2func(cd nua, cd nub, double dm)
{
    if (std::abs(nub - nua) < 1.0e-8)
        return cd(dm, 0.0);
    cd arga = nua * dm;
    cd argb = nub * dm;
    cd exqp = (arga.real() < 40.0) ? std::exp(-arga) : cd(0, 0);
    cd exqq = (argb.real() < 40.0) ? std::exp(-argb) : cd(0, 0);
    return (exqq - exqp) / (nua - nub);
}

// ============================================================
// intijr_one: single integral using already-set s.e / s.einv / s.ra / s.rb
// Call evalg before using this! i0,j0 are 0-based.
// ============================================================
static double intijr_one(int i0, int j0, int m0, int typelyr,
                         double wvno2, RayleighWork& s)
{
    cd cintijr;
    if (s.iwat[m0] == 1) {
        if (typelyr < 0) {
            cd kmpu = s.einv[0][0]*s.uz[m0] + s.einv[0][1]*s.tz[m0];
            cintijr = s.e[i0][0]*s.e[j0][0]*kmpu*kmpu/(2.0*s.ra);
        } else if (typelyr == 0) {
            cd km1pd = s.einv[1][0]*s.uz[m0]   + s.einv[1][1]*s.tz[m0];
            cd kmpu  = s.einv[0][0]*s.uz[m0+1] + s.einv[0][1]*s.tz[m0+1];
            cd FA = ffunc(s.ra, s.zd[m0]);
            cd GA = gfunc(s.ra, s.zd[m0]);
            cintijr = s.e[i0][0]*s.e[j0][0]*kmpu*kmpu*FA
                    + (s.e[i0][0]*s.e[j0][1]+s.e[i0][1]*s.e[j0][0])*kmpu*km1pd*GA
                    + s.e[i0][1]*s.e[j0][1]*km1pd*km1pd*FA;
        } else {
            cd km1pd = s.einv[1][0]*s.uz[m0] + s.einv[1][1]*s.tz[m0];
            cintijr = s.e[i0][1]*s.e[j0][1]*km1pd*km1pd/(2.0*s.ra);
        }
    } else {
        if (typelyr < 0) {
            cd kmpu = s.einv[0][0]*s.ur[m0] + s.einv[0][1]*s.uz[m0]
                    + s.einv[0][2]*s.tz[m0] + s.einv[0][3]*s.tr[m0];
            cd kmsu = s.einv[1][0]*s.ur[m0] + s.einv[1][1]*s.uz[m0]
                    + s.einv[1][2]*s.tz[m0] + s.einv[1][3]*s.tr[m0];
            cintijr = s.e[i0][0]*s.e[j0][0]*kmpu*kmpu/(2.0*s.ra)
                    + (s.e[i0][0]*s.e[j0][1]+s.e[i0][1]*s.e[j0][0])*kmpu*kmsu/(s.ra+s.rb)
                    + s.e[i0][1]*s.e[j0][1]*kmsu*kmsu/(2.0*s.rb);
        } else if (typelyr == 0) {
            cd km1pd = s.einv[2][0]*s.ur[m0]   + s.einv[2][1]*s.uz[m0]
                     + s.einv[2][2]*s.tz[m0]   + s.einv[2][3]*s.tr[m0];
            cd km1sd = s.einv[3][0]*s.ur[m0]   + s.einv[3][1]*s.uz[m0]
                     + s.einv[3][2]*s.tz[m0]   + s.einv[3][3]*s.tr[m0];
            cd kmpu  = s.einv[0][0]*s.ur[m0+1] + s.einv[0][1]*s.uz[m0+1]
                     + s.einv[0][2]*s.tz[m0+1] + s.einv[0][3]*s.tr[m0+1];
            cd kmsu  = s.einv[1][0]*s.ur[m0+1] + s.einv[1][1]*s.uz[m0+1]
                     + s.einv[1][2]*s.tz[m0+1] + s.einv[1][3]*s.tr[m0+1];
            cd FA = ffunc(s.ra, s.zd[m0]);
            cd GA = gfunc(s.ra, s.zd[m0]);
            cd FB = ffunc(s.rb, s.zd[m0]);
            cd GB = gfunc(s.rb, s.zd[m0]);
            cd H1 = h1func(s.ra, s.rb, s.zd[m0]);
            cd H2 = h2func(s.ra, s.rb, s.zd[m0]);
            cintijr = s.e[i0][0]*s.e[j0][0]*kmpu*kmpu*FA
                    + s.e[i0][2]*s.e[j0][2]*km1pd*km1pd*FA
                    + s.e[i0][1]*s.e[j0][1]*kmsu*kmsu*FB
                    + s.e[i0][3]*s.e[j0][3]*km1sd*km1sd*FB
                    + H1*((s.e[i0][0]*s.e[j0][1]+s.e[i0][1]*s.e[j0][0])*kmpu*kmsu
                        + (s.e[i0][2]*s.e[j0][3]+s.e[i0][3]*s.e[j0][2])*km1pd*km1sd)
                    + H2*((s.e[i0][0]*s.e[j0][3]+s.e[i0][3]*s.e[j0][0])*kmpu*km1sd
                        + (s.e[i0][1]*s.e[j0][2]+s.e[i0][2]*s.e[j0][1])*km1pd*kmsu)
                    + GA*(s.e[i0][0]*s.e[j0][2]+s.e[i0][2]*s.e[j0][0])*kmpu*km1pd
                    + GB*(s.e[i0][1]*s.e[j0][3]+s.e[i0][3]*s.e[j0][1])*kmsu*km1sd;
        } else {
            cd km1pd = s.einv[2][0]*s.ur[m0] + s.einv[2][1]*s.uz[m0]
                     + s.einv[2][2]*s.tz[m0] + s.einv[2][3]*s.tr[m0];
            cd km1sd = s.einv[3][0]*s.ur[m0] + s.einv[3][1]*s.uz[m0]
                     + s.einv[3][2]*s.tz[m0] + s.einv[3][3]*s.tr[m0];
            cintijr = s.e[i0][2]*s.e[j0][2]*km1pd*km1pd/(2.0*s.ra)
                    + (s.e[i0][2]*s.e[j0][3]+s.e[i0][3]*s.e[j0][2])*km1pd*km1sd/(s.ra+s.rb)
                    + s.e[i0][3]*s.e[j0][3]*km1sd*km1sd/(2.0*s.rb);
        }
    }
    return std::real(cintijr);
}

// ============================================================
// compute_layer_integrals: call evalg once, return all 6 integrals
// ============================================================
static void compute_layer_integrals(int m0, int typelyr,
                                    double om, double om2, double wvno, double wvno2,
                                    double& INT11, double& INT13, double& INT22,
                                    double& INT24, double& INT33, double& INT44,
                                    RayleighWork& s)
{
    cd gbr[2][5];
    evalg(0, m0, gbr, 0, wvno, om, om2, wvno2, s);

    if (s.iwat[m0] == 1) {
        // fluid: only INT11 and INT22 are used; others zero
        INT11 = intijr_one(0, 0, m0, typelyr, wvno2, s);
        INT22 = intijr_one(1, 1, m0, typelyr, wvno2, s);
        INT13 = 0.0; INT24 = 0.0; INT33 = 0.0; INT44 = 0.0;
    } else {
        INT11 = intijr_one(0, 0, m0, typelyr, wvno2, s);
        INT13 = intijr_one(0, 2, m0, typelyr, wvno2, s);
        INT22 = intijr_one(1, 1, m0, typelyr, wvno2, s);
        INT24 = intijr_one(1, 3, m0, typelyr, wvno2, s);
        INT33 = intijr_one(2, 2, m0, typelyr, wvno2, s);
        INT44 = intijr_one(3, 3, m0, typelyr, wvno2, s);
    }
}

// ============================================================
// getmat: ODE coefficients for layer m0 (0-based)
// ============================================================
static void getmat(int m0, double wvno, double om,
                   double& a12, double& a14, double& a21, double& a23,
                   double& ah, double& av, double& bh, double& bv,
                   double& eta, double& rho,
                   double& TA, double& TC, double& TF, double& TL, double& TN,
                   const RayleighWork& s)
{
    if (s.iwat[m0] == 1) {
        // fluid
        ah  = s.za[m0]; av = s.za[m0];
        bh  = 0.0;       bv = 0.0;
        rho = s.zrho[m0];
        eta = 1.0;

        TL = 0.0;  TN = 0.0;
        TC = s.zrho[m0] * s.za[m0] * s.za[m0];
        TA = TC;
        TF = TA - 2.0 * TN;

        a12 = -(wvno * wvno - om * om / (ah * ah)) / (rho * om * om);
        a14 = 0.0; a21 = 0.0; a23 = 0.0;
    } else {
        // elastic
        ah  = s.za[m0]; av = s.za[m0];
        bh  = s.zb[m0]; bv = s.zb[m0];
        rho = s.zrho[m0];
        eta = 1.0;

        TL = s.zrho[m0] * s.zb[m0] * s.zb[m0];
        TN = TL;
        TC = s.zrho[m0] * s.za[m0] * s.za[m0];
        TA = TC;
        TF = TA - 2.0 * TN;

        a12 = -wvno;
        a14 = 1.0 / TL;
        a21 = wvno * TF / TC;
        a23 = 1.0 / TC;
    }
}

// ============================================================
// energy: compute energy integrals and partial derivatives
// ============================================================
static void energy_func(double om, double wvno, RayleighWork& s)
{
    int mmax = s.mmax;
    double om2   = om * om;
    double wvno2 = wvno * wvno;
    double c     = om / wvno;

    s.sumi0 = 0.0;
    s.sumi1 = 0.0;
    s.sumi2 = 0.0;
    s.sumi3 = 0.0;

    for (int m = 0; m < mmax; m++) {
        double a12, a14, a21, a23, ah, av, bh, bv, eta, rho;
        double TA, TC, TF, TL, TN;
        getmat(m, wvno, om, a12, a14, a21, a23,
               ah, av, bh, bv, eta, rho, TA, TC, TF, TL, TN, s);

        int typelyr = (m == mmax - 1) ? 1 : 0;

        double INT11, INT13, INT22, INT24, INT33, INT44;
        compute_layer_integrals(m, typelyr, om, om2, wvno, wvno2,
                                INT11, INT13, INT22, INT24, INT33, INT44, s);

        if (s.iwat[m] == 1) {
            // fluid
            double URUR   = INT22 * (wvno / (rho * om2)) * (wvno / (rho * om2));
            double UZUZ   = INT11;
            double URDUZ  = -(wvno / (rho * om2)) * a12 * INT22;
            double DUZDUZ = a12 * a12 * INT22;

            s.sumi0 += rho * (URUR + UZUZ);
            s.sumi1 += TA * URUR;
            s.sumi2 += -TF * URDUZ;
            s.sumi3 += TC * DUZDUZ;

            double facah = rho * ah * ah * (URUR - 2.0 * eta * URDUZ / wvno);
            double facav = rho * av * av * DUZDUZ / wvno2;
            s.dcda[m]  = facah + facav;
            double facr = -0.5 * c * c * (URUR + UZUZ);
            s.dcdr[m]  = 0.5 * (av * facav + ah * facah) + facr * rho;
            s.dcdb[m]  = 0.0;
            s.dcdgc[m] = 0.0;
            s.dcdgs[m] = 0.0;

            // ur in fluid from Tz
            s.ur[m] = -wvno * s.tz[m] / (rho * om2);
        } else {
            // solid
            double URUR   = INT11;
            double UZUZ   = INT22;
            double DURDUR = a12*a12*INT22 + 2.0*a12*a14*INT24 + a14*a14*INT44;
            double DUZDUZ = a21*a21*INT11 + 2.0*a21*a23*INT13 + a23*a23*INT33;
            double URDUZ  = a21*INT11 + a23*INT13;
            double UZDUR  = a12*INT22 + a14*INT24;

            s.sumi0 += rho * (URUR + UZUZ);
            s.sumi1 += TL * UZUZ + TA * URUR;
            s.sumi2 += TL * UZDUR - TF * URDUZ;
            s.sumi3 += TL * DURDUR + TC * DUZDUZ;

            double facah = rho * ah * ah * (URUR - 2.0 * eta * URDUZ / wvno);
            double facav = rho * av * av * DUZDUZ / wvno2;
            double facbh = 0.0;
            double facbv = rho * bv * bv * (UZUZ + 2.0*UZDUR/wvno + DURDUR/wvno2
                                            + 4.0*eta*URDUZ/wvno);
            s.dcda[m]  = facah + facav;
            s.dcdb[m]  = facbv + facbh;
            double facr = -0.5 * c * c * (URUR + UZUZ);
            s.dcdr[m]  = 0.5 * (av * facav + ah * facah + bv * facbv) + facr * rho;
            s.dcdgc[m] = rho*bv*bv*(UZUZ + 2.0*UZDUR/wvno + DURDUR/wvno2) + rho*bv*bv*URUR;
            s.dcdgs[m] = s.dcdgc[m];
        }
    }

    // determine final parameters
    s.flagr = om2 * s.sumi0 - wvno2 * s.sumi1 - 2.0 * wvno * s.sumi2 - s.sumi3;
    s.ugr   = (wvno * s.sumi1 + s.sumi2) / (om * s.sumi0);
    s.are   = wvno / (2.0 * om * s.ugr * s.sumi0);
    double fac = s.are * c / wvno2;

    double inv_ug_s0 = 1.0 / (s.ugr * s.sumi0);
    for (int m = 0; m < mmax; m++) {
        s.dcda[m]  *= inv_ug_s0;
        s.dcdb[m]  *= inv_ug_s0;
        s.dcdr[m]  *= inv_ug_s0;
        s.dcdgc[m] *= inv_ug_s0;
        s.dcdgs[m] *= inv_ug_s0;
    }

    // dcdh: thickness partial derivatives
    // (getdcdh computes the dcdh array)
    // called separately as getdcdh_func
    // store fac for getdcdh
    // We call getdcdh inline here:

    // getdcdh
    for (int m = 0; m < mmax; m++) {
        double tuz = s.uz[m];
        double ttz = s.tz[m];
        double ttr = s.tr[m];
        double tur;
        if (s.iwat[m] == 1) {
            double rho_m = s.zrho[m];
            tur = -wvno * ttz / (rho_m * om2);
        } else {
            tur = s.ur[m];
        }

        double gfac1, gfac2, gfac3, gfac4, gfac5, gfac6;
        double drho, dmu, dlm;

        if (m == 0) {
            drho = s.zrho[0];
            dmu  = s.xmu[0];
            dlm  = s.xlam[0];
            double dl2mu = dlm + dmu + dmu;
            double xl2mp = s.xlam[m] + s.xmu[m] + s.xmu[m];

            double duzdzp = (ttz + wvno * s.xlam[m] * tur) / xl2mp;
            double durdzp;
            if (s.iwat[m] == 1)
                durdzp = wvno * tuz;
            else
                durdzp = (ttr / s.xmu[m]) - wvno * tuz;

            double drur2 = tur * tur * drho;
            double dlur2 = tur * tur * dl2mu;

            gfac1 =  om2 * drho * tuz * tuz;
            gfac2 =  om2 * drur2;
            gfac3 = -wvno2 * dmu * tuz * tuz;
            gfac4 = -wvno2 * dlur2;
            gfac5 =  (xl2mp * duzdzp * duzdzp);
            gfac6 =  (s.xmu[m] * durdzp * durdzp);
        } else {
            drho = s.zrho[m] - s.zrho[m-1];
            dmu  = s.xmu[m]  - s.xmu[m-1];
            dlm  = s.xlam[m] - s.xlam[m-1];
            double dl2mu = dlm + dmu + dmu;
            double xl2mp = s.xlam[m]   + s.xmu[m]   + s.xmu[m];
            double xl2mm = s.xlam[m-1] + s.xmu[m-1] + s.xmu[m-1];

            double duzdzp = (ttz + wvno * s.xlam[m] * tur) / xl2mp;
            double durdzp;
            if (s.xmu[m] == 0.0)
                durdzp = wvno * tuz;
            else
                durdzp = (ttr / s.xmu[m]) - wvno * tuz;

            double durdzm;
            if (s.xmu[m-1] == 0.0)
                durdzm = wvno * tuz;
            else
                durdzm = (ttr / s.xmu[m-1]) - wvno * tuz;

            double drur2, dlur2, duzdzm;
            if (s.iwat[m-1] == 1 && s.iwat[m] == 0) {
                double URB = -wvno * s.tz[m] / (s.zrho[m-1] * om2);
                drur2  = tur * tur * s.zrho[m] - URB * URB * s.zrho[m-1];
                dlur2  = tur * tur * xl2mp - URB * URB * xl2mm;
                duzdzm = (ttz + wvno * s.xlam[m-1] * URB) / s.xlam[m-1];
            } else if (s.iwat[m-1] == 1 && s.iwat[m] == 1) {
                double URB = -wvno * s.tz[m] / (s.zrho[m-1] * om2);
                drur2  = tur * tur * s.zrho[m] - URB * URB * s.zrho[m-1];
                dlur2  = tur * tur * xl2mp - URB * URB * xl2mm;
                duzdzm = (ttz + wvno * s.xlam[m-1] * URB) / xl2mm;
            } else {
                drur2  = tur * tur * drho;
                dlur2  = tur * tur * dl2mu;
                duzdzm = (ttz + wvno * s.xlam[m-1] * tur) / xl2mm;
            }

            gfac1 =  om2 * drho * tuz * tuz;
            gfac2 =  om2 * drur2;
            gfac3 = -wvno2 * dmu * tuz * tuz;
            gfac4 = -wvno2 * dlur2;
            gfac5 =  (xl2mp * duzdzp * duzdzp - xl2mm * duzdzm * duzdzm);
            gfac6 =  (s.xmu[m] * durdzp * durdzp - s.xmu[m-1] * durdzm * durdzm);
        }

        double dfac = fac * (gfac1 + gfac2 + gfac3 + gfac4 + gfac5 + gfac6);
        if (std::fabs(dfac) < 1.0e-38) dfac = 0.0;
        s.dcdh[m] = dfac;
    }
}

// ============================================================
// svfunc: combine Dunkin (up) and Haskell (down) eigenfunctions
// ============================================================
static void svfunc(double omega, double wvno, RayleighWork& s)
{
    int mmax = s.mmax;

    double fr;
    up_func(omega, wvno, fr, s);
    down_func(omega, wvno, s);

    // surface values
    double f1213 = -s.cd_arr[0][1];
    s.ur[0] = s.cd_arr[0][2] / s.cd_arr[0][1];
    s.uz[0] = 1.0;
    s.tz[0] = 0.0;
    s.tr[0] = 0.0;

    s.uu0[0] = s.ur[0];
    s.uu0[1] = 1.0;
    s.uu0[2] = fr;
    s.uu0[3] = fr;

    for (int i = 1; i < mmax; i++) {
        double cd1 =  s.cd_arr[i][0];
        double cd2 =  s.cd_arr[i][1];
        double cd3 =  s.cd_arr[i][2];
        double cd4 = -s.cd_arr[i][2];
        double cd5 =  s.cd_arr[i][3];
        double cd6 =  s.cd_arr[i][4];

        double tz1 = -s.vv[i][3];
        double tz2 = -s.vv[i][2];
        double tz3 =  s.vv[i][1];
        double tz4 =  s.vv[i][0];

        double uu1 =  tz2*cd6 - tz3*cd5 + tz4*cd4;
        double uu2 = -tz1*cd6 + tz3*cd3 - tz4*cd2;
        double uu3 =  tz1*cd5 - tz2*cd3 + tz4*cd1;
        double uu4 = -tz1*cd4 + tz2*cd2 - tz3*cd1;

        double ext = s.exa[i] + s.exe[i] - s.exe[0];
        if (ext > -80.0 && ext < 80.0) {
            double fact = std::exp(ext);
            s.ur[i] = uu1 * fact / f1213;
            s.uz[i] = uu2 * fact / f1213;
            s.tz[i] = uu3 * fact / f1213;
            s.tr[i] = uu4 * fact / f1213;
        } else {
            s.ur[i] = 0.0;
            s.uz[i] = 0.0;
            s.tz[i] = 0.0;
            s.tr[i] = 0.0;
        }
    }

    // correction for fluid layers on top (if not all fluid)
    if (!s.allfluid) {
        int jwat = 0;
        for (int i = 0; i < mmax; i++) {
            if (s.iwat[i] > 0)
                jwat = i + 1;  // 1-based count of leading fluid layers
            else
                break;
        }
        if (jwat > 0) {
            for (int i = 0; i < jwat; i++) {
                s.ur[i] = 0.0;
                s.tr[i] = 0.0;
            }
        }
    }
}

// ============================================================
// sprayl: spherical correction for Rayleigh waves
// ============================================================
static void sprayl(double om, double c, double& csph, double& usph,
                   RayleighWork& s)
{
    const double ar = 6371.0;
    double tm  = std::sqrt(1.0 + std::pow(c / (2.0 * ar * om), 2.0));
    double tm3 = tm * tm * tm;

    for (int i = 0; i < s.mmax; i++) {
        s.dcda[i]  *= s.vtp[i] / tm3;
        s.dcdb[i]  *= s.vtp[i] / tm3;
        s.dcdh[i]  *= s.dtp[i] / tm3;
        s.dcdr[i]  *= s.rtp[i] / tm3;
        s.dcdgc[i] *= s.vtp[i] / tm3;
        s.dcdgs[i] *= s.vtp[i] / tm3;
    }

    csph = c    / tm;
    usph = s.ugr * tm;
}

// ============================================================
// suffix_sum: convert depth-derivative to thickness-derivative
// out[i] = sum(in[i+1 .. n-1])
// ============================================================
static void suffix_sum(const double* in, double* out, int n)
{
    double acc = 0.0;
    for (int i = n - 1; i >= 0; i--) {
        out[i] = acc;
        acc   += in[i];
    }
    out[n - 1] = 0.0;
}

// ============================================================
// load_model_r: populate RayleighWork from float arrays
// ============================================================
static void load_model_r(RayleighWork& s, int nlayer,
                         const float* thk, const float* vp,
                         const float* vs, const float* rho)
{
    s.mmax = nlayer;
    s.allfluid = true;
    for (int i = 0; i < nlayer; i++) {
        s.za[i]   = static_cast<double>(vp[i]);
        s.zb[i]   = static_cast<double>(vs[i]);
        s.zrho[i] = static_cast<double>(rho[i]);
        s.zd[i]   = static_cast<double>(thk[i]);
        if (vs[i] > 0.0f) {
            s.allfluid  = false;
            s.iwat[i]   = 0;
        } else {
            s.iwat[i] = 1;
        }
    }
}

// ============================================================
// sregn96_core: compute Rayleigh eigenfunctions
// ============================================================
static RayleighEigenResult sregn96_core(RayleighWork& s,
                                        double period_s,
                                        double phase_vel_km_s,
                                        int nsph)
{
    int nlayer = s.mmax;
    if (nsph > 0) bldsph(s);

    for (int i = 0; i < nlayer; i++) {
        s.xmu[i]  = s.zrho[i] * s.zb[i] * s.zb[i];
        s.xlam[i] = s.zrho[i] * s.za[i] * s.za[i] - 2.0 * s.xmu[i];
    }

    double omega = TWOPI / period_s;
    double c     = phase_vel_km_s;
    double wvno  = omega / c;

    svfunc(omega, wvno, s);
    energy_func(omega, wvno, s);

    double csph, usph;
    if (nsph > 0) {
        sprayl(omega, c, csph, usph, s);
    } else {
        csph = c;
        usph = s.ugr;
    }

    // dcdh: depth -> thickness derivative
    double tmp_dcdh[MAX_LAYERS];
    suffix_sum(s.dcdh, tmp_dcdh, nlayer);

    RayleighEigenResult res;
    res.cp = csph;
    res.cg = usph;
    res.dispu  .assign(s.ur,     s.ur     + nlayer);
    res.dispw  .assign(s.uz,     s.uz     + nlayer);
    res.stressu.assign(s.tr,     s.tr     + nlayer);
    res.stressw.assign(s.tz,     s.tz     + nlayer);
    res.dc2da  .assign(s.dcda,   s.dcda   + nlayer);
    res.dc2db  .assign(s.dcdb,   s.dcdb   + nlayer);
    res.dc2dh  .assign(tmp_dcdh, tmp_dcdh + nlayer);
    res.dc2dr  .assign(s.dcdr,   s.dcdr   + nlayer);
    return res;
}

// ============================================================
// sregn96_hti_core: same + HTI kernels
// ============================================================
static HTIResult sregn96_hti_core(RayleighWork& s,
                                  double period_s,
                                  double phase_vel_km_s,
                                  int nsph)
{
    int nlayer = s.mmax;
    if (nsph > 0) bldsph(s);

    for (int i = 0; i < nlayer; i++) {
        s.xmu[i]  = s.zrho[i] * s.zb[i] * s.zb[i];
        s.xlam[i] = s.zrho[i] * s.za[i] * s.za[i] - 2.0 * s.xmu[i];
    }

    double omega = TWOPI / period_s;
    double c     = phase_vel_km_s;
    double wvno  = omega / c;

    svfunc(omega, wvno, s);
    energy_func(omega, wvno, s);

    double csph, usph;
    if (nsph > 0) {
        sprayl(omega, c, csph, usph, s);
    } else {
        csph = c;
        usph = s.ugr;
    }

    double tmp_dcdh[MAX_LAYERS];
    suffix_sum(s.dcdh, tmp_dcdh, nlayer);

    HTIResult res;
    res.cp = csph;
    res.cg = usph;
    res.dispu  .assign(s.ur,     s.ur     + nlayer);
    res.dispw  .assign(s.uz,     s.uz     + nlayer);
    res.stressu.assign(s.tr,     s.tr     + nlayer);
    res.stressw.assign(s.tz,     s.tz     + nlayer);
    res.dc2da  .assign(s.dcda,   s.dcda   + nlayer);
    res.dc2db  .assign(s.dcdb,   s.dcdb   + nlayer);
    res.dc2dh  .assign(tmp_dcdh, tmp_dcdh + nlayer);
    res.dc2dr  .assign(s.dcdr,   s.dcdr   + nlayer);
    res.dc2dgc .assign(s.dcdgc,  s.dcdgc  + nlayer);
    res.dc2dgs .assign(s.dcdgs,  s.dcdgs  + nlayer);
    return res;
}

// ============================================================
// sregnpu_core: compute group-velocity kernels
// ============================================================
static RayleighGroupKernelResult sregnpu_core(RayleighWork& s,
    double t,  double cp,
    double t1, double cp1,
    double t2, double cp2,
    int nsph)
{
    int nlayer = s.mmax;
    if (nsph > 0) bldsph(s);

    for (int i = 0; i < nlayer; i++) {
        s.xmu[i]  = s.zrho[i] * s.zb[i] * s.zb[i];
        s.xlam[i] = s.zrho[i] * s.za[i] * s.za[i] - 2.0 * s.xmu[i];
    }

    // --- Step 1: (t, cp) ---
    double omega = TWOPI / t;
    double wvno  = omega / cp;
    svfunc(omega, wvno, s);
    energy_func(omega, wvno, s);
    double cg_t = s.ugr;
    std::memcpy(s.dcda_t, s.dcda, nlayer * sizeof(double));
    std::memcpy(s.dcdb_t, s.dcdb, nlayer * sizeof(double));
    std::memcpy(s.dcdr_t, s.dcdr, nlayer * sizeof(double));
    std::memcpy(s.dcdh_t, s.dcdh, nlayer * sizeof(double));
    std::memcpy(s.ur_t,   s.ur,   nlayer * sizeof(double));
    std::memcpy(s.uz_t,   s.uz,   nlayer * sizeof(double));
    std::memcpy(s.tz_t,   s.tz,   nlayer * sizeof(double));
    std::memcpy(s.tr_t,   s.tr,   nlayer * sizeof(double));

    // --- Step 2: (t1, cp1) ---
    omega = TWOPI / t1;
    wvno  = omega / cp1;
    svfunc(omega, wvno, s);
    energy_func(omega, wvno, s);
    std::memcpy(s.dcda1, s.dcda, nlayer * sizeof(double));
    std::memcpy(s.dcdb1, s.dcdb, nlayer * sizeof(double));
    std::memcpy(s.dcdr1, s.dcdr, nlayer * sizeof(double));
    std::memcpy(s.dcdh1, s.dcdh, nlayer * sizeof(double));

    // --- Step 3: (t2, cp2) ---
    omega = TWOPI / t2;
    wvno  = omega / cp2;
    svfunc(omega, wvno, s);
    energy_func(omega, wvno, s);
    std::memcpy(s.dcda2, s.dcda, nlayer * sizeof(double));
    std::memcpy(s.dcdb2, s.dcdb, nlayer * sizeof(double));
    std::memcpy(s.dcdr2, s.dcdr, nlayer * sizeof(double));
    std::memcpy(s.dcdh2, s.dcdh, nlayer * sizeof(double));

    // Restore t state
    std::memcpy(s.dcda, s.dcda_t, nlayer * sizeof(double));
    std::memcpy(s.dcdb, s.dcdb_t, nlayer * sizeof(double));
    std::memcpy(s.dcdr, s.dcdr_t, nlayer * sizeof(double));
    std::memcpy(s.dcdh, s.dcdh_t, nlayer * sizeof(double));
    std::memcpy(s.ur,   s.ur_t,   nlayer * sizeof(double));
    std::memcpy(s.uz,   s.uz_t,   nlayer * sizeof(double));
    std::memcpy(s.tz,   s.tz_t,   nlayer * sizeof(double));
    std::memcpy(s.tr,   s.tr_t,   nlayer * sizeof(double));
    s.ugr = cg_t;

    double uc1     = cg_t / cp;
    double uc1_a   = uc1 * (2.0 - uc1);
    double uc1_b   = uc1 * uc1 * t / (t2 - t1);

    double du2da[MAX_LAYERS], du2db[MAX_LAYERS];
    double du2dr[MAX_LAYERS], du2dh[MAX_LAYERS];
    for (int i = 0; i < nlayer; i++) {
        du2da[i] = uc1_a*s.dcda2[i] - uc1_b*(s.dcda2[i] - s.dcda1[i]);
        du2db[i] = uc1_a*s.dcdb2[i] - uc1_b*(s.dcdb2[i] - s.dcdb1[i]);
        du2dr[i] = uc1_a*s.dcdr2[i] - uc1_b*(s.dcdr2[i] - s.dcdr1[i]);
        du2dh[i] = uc1_a*s.dcdh2[i] - uc1_b*(s.dcdh2[i] - s.dcdh1[i]);
    }

    double csph = cp, usph = cg_t;

    if (nsph > 0) {
        const double ar = 6371.0;
        omega = TWOPI / t;
        double tm  = std::sqrt(1.0 + std::pow(cp / (2.0 * ar * omega), 2.0));
        double tm3 = tm * tm * tm;
        double tm1 = std::pow(0.5 / (ar * omega), 2.0) / tm;

        for (int i = 0; i < nlayer; i++) {
            du2da[i] = (tm * du2da[i] + cg_t * cp * s.dcda[i] * tm1) * s.vtp[i];
            du2db[i] = (tm * du2db[i] + cg_t * cp * s.dcdb[i] * tm1) * s.vtp[i];
            du2dr[i] = (tm * du2dr[i] + cg_t * cp * s.dcdr[i] * tm1) * s.rtp[i];
            du2dh[i] = (tm * du2dh[i] + cg_t * cp * s.dcdh[i] * tm1) * s.dtp[i];
            s.dcda[i] = s.dcda[i] / tm3 * s.vtp[i];
            s.dcdb[i] = s.dcdb[i] / tm3 * s.vtp[i];
            s.dcdr[i] = s.dcdr[i] / tm3 * s.rtp[i];
            s.dcdh[i] = s.dcdh[i] / tm3 * s.dtp[i];
        }
        csph = cp   / tm;
        usph = cg_t * tm;
    }

    double tmp_dcdh[MAX_LAYERS], tmp_du2dh[MAX_LAYERS];
    suffix_sum(s.dcdh, tmp_dcdh,  nlayer);
    suffix_sum(du2dh,  tmp_du2dh, nlayer);

    RayleighGroupKernelResult res;
    res.cp = csph;
    res.cg = usph;
    res.dispu  .assign(s.ur,     s.ur     + nlayer);
    res.dispw  .assign(s.uz,     s.uz     + nlayer);
    res.stressu.assign(s.tr,     s.tr     + nlayer);
    res.stressw.assign(s.tz,     s.tz     + nlayer);
    res.dc2da  .assign(s.dcda,   s.dcda   + nlayer);
    res.dc2db  .assign(s.dcdb,   s.dcdb   + nlayer);
    res.dc2dh  .assign(tmp_dcdh, tmp_dcdh + nlayer);
    res.dc2dr  .assign(s.dcdr,   s.dcdr   + nlayer);
    res.du2da  .assign(du2da,    du2da    + nlayer);
    res.du2db  .assign(du2db,    du2db    + nlayer);
    res.du2dh  .assign(tmp_du2dh,tmp_du2dh+ nlayer);
    res.du2dr  .assign(du2dr,    du2dr    + nlayer);
    return res;
}

// ============================================================
// Public API implementations
// ============================================================

// --- sregn96 vector<Layer> ---
RayleighEigenResult rayleighPhaseKernel(
    const std::vector<Layer>& model,
    double period_s, double phase_vel_km_s,
    EarthModel earth)
{
    int nlayer = static_cast<int>(model.size());
    if (nlayer > MAX_LAYERS)
        throw std::invalid_argument("sregn96: too many layers");

    RayleighWork& s = tl_work;
    s.mmax = nlayer;
    s.allfluid = true;
    for (int i = 0; i < nlayer; i++) {
        s.za[i]   = static_cast<double>(model[i].vp);
        s.zb[i]   = static_cast<double>(model[i].vs);
        s.zrho[i] = static_cast<double>(model[i].density);
        s.zd[i]   = static_cast<double>(model[i].thickness);
        if (model[i].vs > 0.0f) { s.allfluid = false; s.iwat[i] = 0; }
        else                     { s.iwat[i] = 1; }
    }
    return sregn96_core(s, period_s, phase_vel_km_s,
                        (earth == EarthModel::Spherical) ? 1 : 0);
}

// --- sregn96 array ---
RayleighEigenResult rayleighPhaseKernel(
    int nlayer,
    const float* thk, const float* vp, const float* vs, const float* rho,
    double period_s, double phase_vel_km_s,
    EarthModel earth)
{
    if (nlayer > MAX_LAYERS)
        throw std::invalid_argument("sregn96: too many layers");
    RayleighWork& s = tl_work;
    load_model_r(s, nlayer, thk, vp, vs, rho);
    return sregn96_core(s, period_s, phase_vel_km_s,
                        (earth == EarthModel::Spherical) ? 1 : 0);
}

// --- sregn96_hti vector<Layer> ---
HTIResult rayleighPhaseKernel_hti(
    const std::vector<Layer>& model,
    double period_s, double phase_vel_km_s,
    EarthModel earth)
{
    int nlayer = static_cast<int>(model.size());
    if (nlayer > MAX_LAYERS)
        throw std::invalid_argument("sregn96_hti: too many layers");

    RayleighWork& s = tl_work;
    s.mmax = nlayer;
    s.allfluid = true;
    for (int i = 0; i < nlayer; i++) {
        s.za[i]   = static_cast<double>(model[i].vp);
        s.zb[i]   = static_cast<double>(model[i].vs);
        s.zrho[i] = static_cast<double>(model[i].density);
        s.zd[i]   = static_cast<double>(model[i].thickness);
        if (model[i].vs > 0.0f) { s.allfluid = false; s.iwat[i] = 0; }
        else                     { s.iwat[i] = 1; }
    }
    return sregn96_hti_core(s, period_s, phase_vel_km_s,
                            (earth == EarthModel::Spherical) ? 1 : 0);
}

// --- sregn96_hti array ---
HTIResult rayleighPhaseKernel_hti(
    int nlayer,
    const float* thk, const float* vp, const float* vs, const float* rho,
    double period_s, double phase_vel_km_s,
    EarthModel earth)
{
    if (nlayer > MAX_LAYERS)
        throw std::invalid_argument("sregn96_hti: too many layers");
    RayleighWork& s = tl_work;
    load_model_r(s, nlayer, thk, vp, vs, rho);
    return sregn96_hti_core(s, period_s, phase_vel_km_s,
                            (earth == EarthModel::Spherical) ? 1 : 0);
}

// --- sregnpu vector<Layer> ---
RayleighGroupKernelResult rayleighGroupKernel(
    const std::vector<Layer>& model,
    double t, double cp, double t1, double cp1, double t2, double cp2,
    EarthModel earth)
{
    int nlayer = static_cast<int>(model.size());
    if (nlayer > MAX_LAYERS)
        throw std::invalid_argument("sregnpu: too many layers");

    RayleighWork& s = tl_work;
    s.mmax = nlayer;
    s.allfluid = true;
    for (int i = 0; i < nlayer; i++) {
        s.za[i]   = static_cast<double>(model[i].vp);
        s.zb[i]   = static_cast<double>(model[i].vs);
        s.zrho[i] = static_cast<double>(model[i].density);
        s.zd[i]   = static_cast<double>(model[i].thickness);
        if (model[i].vs > 0.0f) { s.allfluid = false; s.iwat[i] = 0; }
        else                     { s.iwat[i] = 1; }
    }
    return sregnpu_core(s, t, cp, t1, cp1, t2, cp2,
                        (earth == EarthModel::Spherical) ? 1 : 0);
}

// --- sregnpu array ---
RayleighGroupKernelResult rayleighGroupKernel(
    int nlayer,
    const float* thk, const float* vp, const float* vs, const float* rho,
    double t, double cp, double t1, double cp1, double t2, double cp2,
    EarthModel earth)
{
    if (nlayer > MAX_LAYERS)
        throw std::invalid_argument("sregnpu: too many layers");
    RayleighWork& s = tl_work;
    load_model_r(s, nlayer, thk, vp, vs, rho);
    return sregnpu_core(s, t, cp, t1, cp1, t2, cp2,
                        (earth == EarthModel::Spherical) ? 1 : 0);
}
