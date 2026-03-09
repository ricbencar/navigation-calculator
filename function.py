# ==============================================================================
#  function.py
#  ------------------------------------------------------------------------------
#  Standalone + importable wavelength calculator using the same solver core as
#  fenton_gui.py (Fenton stream-function / Fourier method, finite depth, period
#  input, Eulerian current criterion).
#
#  External API (for other programs):
#      import fenton   # rename this file to fenton.py, or import via your package
#      L = fenton.L(H, T, d, U)
#
#  CLI usage:
#      python function.py                 # interactive prompts (defaults: 3, 9, 5, 1)
#      python function.py H T d U         # prints wavelength to console
# ==============================================================================

from __future__ import annotations

import os
import tempfile
import warnings
import numpy as np

# ------------------------------------------------------------------------------
#  RUNTIME STABILITY (threads / BLAS)
# ------------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Ensure Numba cache is writable on Windows (avoids slowdowns/permission issues).
if os.name == "nt":
    try:
        _numba_cache_dir = os.path.join(tempfile.gettempdir(), "fenton_numba_cache")
        os.makedirs(_numba_cache_dir, exist_ok=True)
        os.environ.setdefault("NUMBA_CACHE_DIR", _numba_cache_dir)
    except Exception:
        pass

# ==============================================================================
#  NUMPY/NUMBA COMPAT
# ==============================================================================
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

# ==============================================================================
#  GLOBAL CONSTANTS (matched to fenton_gui.py)
# ==============================================================================
G_STD = 9.80665         # Standard Gravity [m/s^2]
RHO = 1025.0          # Density of Seawater [kg/m^3]
DTYPE = np.float64      # Precision for floating point arithmetic
N_FOURIER = 50          # Order of Fourier Series (N=50)

warnings.filterwarnings("ignore")


def _np_trapz(y, x):
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x)
    trapz = getattr(np, "trapz", None)
    if trapz is not None:
        return trapz(y, x)
    # Minimal fallback (should never trigger on normal NumPy builds)
    y = np.asarray(y)
    x = np.asarray(x)
    if y.size < 2:
        return 0.0
    dx = np.diff(x)
    return np.sum((y[:-1] + y[1:]) * 0.5 * dx)

@njit(cache=True)
def _eqns_numba(
    z,
    rhs,
    coeff,
    Tanh,
    cos_nm,
    sin_nm,
    n,
    num,
    Hoverd,
    height,
    Current,
    Current_criterion,
):
    """Numba implementation of Eqns() (finite depth, Period case) with 1-based indexing."""
    pi = np.pi

    # Eqn 1
    rhs[1] = z[2] - z[1] * Hoverd

    # Eqn 2 (Period case)
    rhs[2] = z[2] - height * z[3] * z[3]

    # Eqn 3
    rhs[3] = z[4] * z[3] - 2.0 * pi

    # Eqn 4
    rhs[4] = z[5] + z[7] - z[4]

    # Eqn 5
    rhs[5] = z[1] * (z[6] + z[7] - z[4]) - z[8]

    # coeff and tanh tables
    kd = z[1]
    for i in range(1, n + 1):
        coeff[i] = z[n + i + 10]
        Tanh[i] = np.tanh(i * kd)

    # Eqn 6 (finite depth; correction uses sqrt(z[1]))
    rhs[6] = z[Current_criterion + 4] - Current * np.sqrt(kd)

    # Eqn 7 (mean free surface level; scaling constant irrelevant)
    rhs7 = z[10] + z[n + 10]
    for i in range(1, n):
        rhs7 += 2.0 * z[10 + i]
    rhs[7] = rhs7

    # Eqn 8 (wave height definition)
    rhs[8] = z[10] - z[n + 10] - z[2]

    # Eqns 9..(n+9) and (n+10)..(2n+10): free-surface BCs
    for m in range(0, n + 1):
        zsurf = z[10 + m]  # k(eta-d) at this node

        psi = 0.0
        u = 0.0
        v = 0.0

        for jj in range(1, n + 1):
            cj = coeff[jj]
            tj = Tanh[jj]

            x = jj * zsurf
            # Prevent overflow if the iteration diverges (numerical safeguard)
            if x > 60.0 or x < -60.0:
                rhs[1] = np.inf
                return np.inf
            e = np.exp(x)
            inv_e = 1.0 / e
            sinhkd = 0.5 * (e - inv_e)
            coshkd = 0.5 * (e + inv_e)

            # Hyperbolic rewrite (A-8): C = cosh + sinh*tanh(jkd), S = sinh + cosh*tanh(jkd)
            S = sinhkd + coshkd * tj
            C = coshkd + sinhkd * tj

            ccos = cos_nm[m, jj - 1]
            ssin = sin_nm[m, jj - 1]

            psi += cj * S * ccos
            jcj = jj * cj
            u += jcj * C * ccos
            v += jcj * S * ssin

        rhs[m + 9] = psi - z[8] - z[7] * z[m + 10]
        rhs[n + m + 10] = 0.5 * ((-z[7] + u) ** 2 + v * v) + z[m + 10] - z[9]

    # Sum of squares
    ss = 0.0
    for i in range(1, num + 1):
        ss += rhs[i] * rhs[i]
    return ss


@njit(cache=True)
def _compute_Y_and_B_numba(z, B, Y, cosa, n):
    """Numba implementation of the post-convergence Fourier transform block."""
    for i in range(0, Y.shape[0]):
        Y[i] = 0.0

    two_n = 2 * n

    for j in range(1, n + 1):
        B[j] = z[j + n + 10]

        # s = 0.5 * (z[10] + z[n+10] * ((-1.0)**j))
        sign = 1.0
        if (j % 2) == 1:
            sign = -1.0
        s = 0.5 * (z[10] + z[n + 10] * sign)

        for m in range(1, n):
            s += z[10 + m] * cosa[(m * j) % two_n]

        Y[j] = 2.0 * s / n


@njit(cache=True)
def _surface_keta_numba(Y, n, X):
    """Numba implementation of Surface(X): returns k(eta-d) at phase X."""
    kEta = 0.0
    for j in range(1, n):
        kEta += Y[j] * np.cos(j * X)
    kEta += 0.5 * Y[n] * np.cos(n * X)
    return kEta


@njit(cache=True)
def _point_numba(X, Y, z, Tanh, B, n):
    """Numba implementation of the finite-depth Point(X,Y) kernel."""
    kd = z[1]

    # depth-scaled dimensionless bulk values
    c = z[4] / np.sqrt(kd)
    ce = z[5] / np.sqrt(kd)
    R = 1.0 + z[9] / kd

    # local variables in wave scaling
    u = 0.0
    v = 0.0
    ux = 0.0
    vx = 0.0

    for j in range(1, n + 1):
        Cos = np.cos(j * X)
        Sin = np.sin(j * X)

        coshdelta = np.cosh(j * Y)
        sinhdelta = np.sinh(j * Y)
        C = coshdelta + sinhdelta * Tanh[j]
        S = sinhdelta + coshdelta * Tanh[j]

        Bj = B[j]
        u += j * Bj * C * Cos
        v += j * Bj * S * Sin
        ux += -(j * j) * Bj * C * Sin
        vx += (j * j) * Bj * S * Cos

    # convert to depth scaling (see C++ comments)
    inv_kd_sqrt = 1.0 / np.sqrt(kd)
    inv_kd_32 = 1.0 / (kd ** 1.5)

    u *= inv_kd_sqrt
    v *= inv_kd_sqrt
    ux *= np.sqrt(kd)
    vx *= np.sqrt(kd)

    # add Euler current to u
    u = ce + u

    # time derivatives (steady in moving frame)
    ut = -c * ux
    vt = -c * vx
    uy = vx
    vy = -ux

    dudt = ut + u * ux + v * uy

    return float(u), float(v), float(dudt)


class FentonStreamFunction:
    """
    Fenton steady-wave solver using the Fourier approximation / stream-function
    method, implemented as a direct port of the provided C++ reference code.

    Public API is preserved to keep the GUI unchanged.
    """

    # --------------------------- construction ---------------------------------

    def __init__(self, H, T, d, U=0.0):
        # Inputs (physical)
        self.H_target = float(H)    # [m]
        self.T_target = float(T)    # [s]
        self.d = float(d)    # [m]
        self.U = float(U)    # [m/s] (Eulerian / lab-frame)

        # Constants
        self.g = G_STD
        self.N = N_FOURIER

        # Solver control (C++-style defaults)
        self.nstep = 4        # continuation steps in wave height
        self.number = 40       # max Newton iterations per step
        self.crit = 1.0e-8   # intermediate-step convergence factor (C++: crit)
        self.criter_final = 1.0e-10  # final-step convergence factor

        # Problem mode (matches GUI: finite depth, Period input)
        self.Depth = "Finite"
        self.Case = "Period"

        # Current criterion: 1=Eulerian, 2=Stokes (GUI input is Eulerian current)
        self.Current_criterion = 1

        # Derived input non-dimensional groups (C++ Read_data equivalents)
        # MaxH == H/d;  T_nd == T*sqrt(g/d); Height == (H/d)/(T_nd^2) == H/(g T^2)
        self.MaxH = self.H_target / self.d if self.d > 0 else 0.0
        self.T_nd = self.T_target * np.sqrt(self.g / self.d) if self.d > 0 else 0.0
        self.Height = (self.MaxH / (self.T_nd * self.T_nd)) if self.T_nd > 0 else 0.0

        # Current input in C++ is dimensionless w.r.t. sqrt(g d) for finite depth
        self.Current = self.U / np.sqrt(self.g * self.d) if self.d > 0 else 0.0

        # ------------------------- outputs (public) ----------------------------
        self.k = 0.0
        self.L = 0.0
        self.c = 0.0
        self.converged = False


        # Human-readable failure reason (used by GUI when convergence fails)
        self.last_error = ""

        # Robustness for large ambient currents: allow more continuation steps and
        # Newton iterations. This does NOT change equations; it only increases the
        # solver budget to avoid premature failure when |U| is high.
        if abs(self.Current) >= 1.0:
            self.nstep = max(self.nstep, 8)
            self.number = max(self.number, 80)
        self.Bj = np.zeros(self.N, dtype=DTYPE)             # B_1..B_N (0-based in Python)
        self.eta_nodes = np.zeros(self.N + 1, dtype=DTYPE)  # absolute z from bed [m]

        self.eta_crest = 0.0
        self.eta_trough = 0.0
        self.steepness = 0.0
        self.rel_depth = 0.0
        self.ursell = 0.0
        self.regime = ""

        self.breaking_index = 0.0
        self.is_breaking = False
        self.breaking_limit_miche = 0.0

        self.u_bed = 0.0
        self.tau_bed = 0.0
        self.acc_max = 0.0
        self.u_surf = 0.0
        self.w_max = 0.0
        self.asymmetry = 0.0
        self.ExcursionBed = 0.0

        self.Cg = 0.0
        self.Power = 0.0
        self.EnergyDensity = 0.0
        self.Sxx = 0.0
        self.Impulse = 0.0
        self.MassTransport = 0.0
        self.BernoulliR = 0.0

        # ------------------------- internal C++ arrays -------------------------
        self.n = int(self.N)
        self.num = 2 * self.n + 10

        # 1-based vectors (index 0 unused)
        self.z = np.zeros(self.num + 1, dtype=DTYPE)
        self.rhs1 = np.zeros(self.num + 1, dtype=DTYPE)
        self.rhs2 = np.zeros(self.num + 1, dtype=DTYPE)
        self.coeff = np.zeros(self.n + 1, dtype=DTYPE)    # coeff[1..n]
        self.Tanh = np.zeros(self.n + 1, dtype=DTYPE)    # Tanh[1..n]
        self.B = np.zeros(self.n + 1, dtype=DTYPE)    # B[1..n]
        self.Y = np.zeros(self.num + 1, dtype=DTYPE)  # Y[0..n] used; keep size

        # Precomputed trig tables as in init()
        self.cosa = np.zeros(2 * self.n + 1, dtype=DTYPE)  # [0..2n]
        self.sina = np.zeros(2 * self.n + 1, dtype=DTYPE)

        # Precompute constant trig tables and collocation lookup tables (C++ init())
        k_idx = np.arange(0, 2 * self.n + 1, dtype=DTYPE)
        self.cosa[:] = np.cos(k_idx * np.pi / self.n)
        self.sina[:] = np.sin(k_idx * np.pi / self.n)

        self._j = np.arange(1, self.n + 1, dtype=DTYPE)
        self._j_int = np.arange(1, self.n + 1, dtype=np.int64)
        self._nm_map = (
            np.arange(0, self.n + 1, dtype=np.int64)[:, None]
            * self._j_int[None, :]
        ) % (2 * self.n)
        self._cos_nm = self.cosa[self._nm_map]  # shape (n+1, n)
        self._sin_nm = self.sina[self._nm_map]

        # Extrapolation storage sol[i][1..2]
        self.sol = np.zeros((self.num + 1, 3), dtype=DTYPE)

        # Run-time step variables (C++ globals)
        self.height = 0.0    # stepped 'height' (dimensionless)
        self.Hoverd = 0.0    # stepped H/d

    # --------------------------- C++ port helpers -----------------------------

    def _init_linear(self):
        """
        Port of C++ init() for finite-depth, Period case (with current criterion).
        Produces an initial state in z[1..num] for the first height step.
        """
        n = self.n
        pi = np.pi

        # For finite depth
        sigma = 2.0 * pi * np.sqrt(self.height / self.Hoverd) if self.Hoverd > 0 else 0.0

        # Fenton & McKee (1990) approximation used in the C++ (commented alternatives omitted)
        if sigma > 0:
            self.z[1] = (sigma * sigma) / (np.tanh(sigma ** 1.5) ** (2.0 / 3.0))
        else:
            # very small waves / degenerate: start with something benign
            self.z[1] = 2.0 * pi * max(self.height, 1e-12) / max(self.Hoverd, 1e-12)

        self.z[2] = self.z[1] * self.Hoverd
        self.z[4] = np.sqrt(np.tanh(self.z[1]))
        self.z[3] = 2.0 * pi / self.z[4]

        # Current initialisation (finite)
        if self.Current_criterion == 1:
            self.z[5] = self.Current * np.sqrt(self.z[2])
            self.z[6] = 0.0
        else:
            self.z[6] = self.Current * np.sqrt(self.z[2])
            self.z[5] = 0.0

        self.z[7] = self.z[4]
        self.z[8] = 0.0
        self.z[9] = 0.5 * self.z[7] * self.z[7]

        # Initial surface elevation nodes and Fourier coefficients (B_j)
        self.z[10] = 0.5 * self.z[2]
        for i in range(1, n + 1):
            self.z[n + i + 10] = 0.0
            self.z[i + 10] = 0.5 * self.z[2] * self.cosa[i]

        self.z[n + 11] = 0.5 * self.z[2] / self.z[7]

        # store sol[] for extrapolation (C++ sets sol[10..] to zero for very first)
        for i in range(1, 10):
            self.sol[i, 1] = self.z[i]
        for i in range(10, self.num + 1):
            self.sol[i, 1] = 0.0

    def _eqns(self, rhs_out):
        """
        Port of C++ Eqns(double *rhs). Fills rhs_out[1..num] and returns sum(rhs^2).
        Finite-depth branch only (GUI mode).
        """
        # Numba-accelerated kernel (drops back to pure NumPy if Numba is unavailable).
        #
        # Robustness note:
        # If the JIT path produces NaN/Inf (usually due to a diverging Newton iterate),
        # fall back to the pure-NumPy path below (same algebra, clearer exceptions).
        if NUMBA_AVAILABLE:
            ss = _eqns_numba(self.z, rhs_out, self.coeff, self.Tanh, self._cos_nm, self._sin_nm,
                self.n,
                self.num,
                self.Hoverd,
                self.height,
                self.Current,
                self.Current_criterion,
            )
            if np.isfinite(ss) and np.isfinite(rhs_out[1:self.num + 1]).all():
                return ss
            # else: continue into the NumPy implementation

        n = self.n
        num = self.num
        pi = np.pi
        z = self.z
        rhs = rhs_out

        # Eqn 1
        rhs[1] = z[2] - z[1] * self.Hoverd

        # Eqn 2 (Period case)
        rhs[2] = z[2] - self.height * z[3] * z[3]

        # Eqn 3
        rhs[3] = z[4] * z[3] - 2.0 * pi

        # Eqn 4
        rhs[4] = z[5] + z[7] - z[4]

        # Eqn 5
        rhs[5] = z[1] * (z[6] + z[7] - z[4]) - z[8]

        # coeff and tanh tables
        for i in range(1, n + 1):
            self.coeff[i] = z[n + i + 10]
            self.Tanh[i] = np.tanh(i * z[1])

        # Eqn 6 (finite depth; correction uses sqrt(z[1]))
        rhs[6] = z[self.Current_criterion + 4] - self.Current * np.sqrt(z[1])

        # Eqn 7 (mean free surface level; scaling constant irrelevant)
        rhs[7] = z[10] + z[n + 10]
        for i in range(1, n):
            rhs[7] += 2.0 * z[10 + i]

        # Eqn 8 (wave height definition)
        rhs[8] = z[10] - z[n + 10] - z[2]

        # Eqns 9..(n+9) and (n+10)..(2n+10): free-surface BCs
        j = self._j                      # shape (n,)
        coeff = self.coeff[1:n + 1]      # shape (n,)
        tanh = self.Tanh[1:n + 1]        # shape (n,)
        jcoeff = j * coeff               # shape (n,)

        for m in range(0, n + 1):
            zsurf = z[10 + m]  # k(eta-d) at this node

            x = j * zsurf
            if np.any(x > 60.0) or np.any(x < -60.0):
                raise FloatingPointError("Divergence: exp(j*zsurf) out of safe range.")
            e = np.exp(x)
            inv_e = 1.0 / e
            sinhkd = 0.5 * (e - inv_e)
            coshkd = 0.5 * (e + inv_e)

            # Hyperbolic rewrite (A-8): C = cosh + sinh*tanh(jkd), S = sinh + cosh*tanh(jkd)
            S = sinhkd + coshkd * tanh
            C = coshkd + sinhkd * tanh

            cosnm = self._cos_nm[m]
            sinnm = self._sin_nm[m]

            psi = float(np.sum(coeff * S * cosnm))
            u = float(np.sum(jcoeff * C * cosnm))
            v = float(np.sum(jcoeff * S * sinnm))

            rhs[m + 9] = psi - z[8] - z[7] * z[m + 10]
            rhs[n + m + 10] = 0.5 * ((-z[7] + u) ** 2 + v * v) + z[m + 10] - z[9]

        return float(np.dot(rhs[1:num + 1], rhs[1:num + 1]))

    @staticmethod
    def _svd_solve(A, b):
        """
        Solve A x = b via SVD with Press et al. truncation:
          wmin = wmax * 1e-12

        Numerical safety:
        - Reject NaNs/Infs before entering LAPACK.
        - Fallback to least-squares if SVD fails to converge.
        """
        if (not np.isfinite(A).all()) or (not np.isfinite(b).all()):
            raise FloatingPointError("Non-finite values in Jacobian system (A or b).")

        try:
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            smax = np.max(s) if s.size else 0.0
            wmin = smax * 1.0e-12

            s_inv = np.zeros_like(s)
            mask = s > wmin
            s_inv[mask] = 1.0 / s[mask]

            return (Vt.T @ (s_inv * (U.T @ b)))

        except np.linalg.LinAlgError:
            # Conservative fallback: least-squares solution of the same linearised system.
            x, *_ = np.linalg.lstsq(A, b, rcond=1.0e-12)
            return x



    def _newton(self, iter_count):
        """
        Port of the C++ Newton(...) update with additional damping safeguards.

        The governing residual equations are unchanged. The only additions are:
        - Finite-difference step clamping (prevents extreme perturbations if z[i] diverges).
        - Backtracking line-search (reduces step when a full Newton step increases residuals).
        """
        n = self.n
        num = self.num

        # baseline residual
        ss0 = float(self._eqns(self.rhs1))
        if not np.isfinite(ss0):
            raise FloatingPointError("Non-finite residual norm at start of Newton step.")

        z0 = self.z.copy()

        A = np.zeros((num, num), dtype=DTYPE)
        b = np.zeros((num,), dtype=DTYPE)

        # finite-difference Jacobian (column-wise)
        for i in range(1, num + 1):
            h = 0.01 * z0[i]
            if abs(z0[i]) < 1.0e-4:
                h = 1.0e-5
            # clamp perturbation magnitude (purely numerical safeguard)
            if abs(h) > 1.0:
                h = np.copysign(1.0, h)

            self.z[i] = z0[i] + h
            self._eqns(self.rhs2)
            self.z[i] = z0[i]

            b[i - 1] = -self.rhs1[i]
            A[:, i - 1] = (self.rhs2[1:num + 1] - self.rhs1[1:num + 1]) / h

        dx = self._svd_solve(A, b)
        if not np.isfinite(dx).all():
            raise FloatingPointError("Non-finite Newton correction vector (dx).")

        # Backtracking: prefer alpha=1, reduce if it worsens residuals or violates kd>0
        alpha = 1.0
        ss_best = ss0
        z_best = z0

        while alpha >= 1.0e-4:
            z_try = z0.copy()
            z_try[1:num + 1] = z0[1:num + 1] + alpha * dx

            # Must keep kd positive and all values finite
            if (z_try[1] <= 0.0) or (not np.isfinite(z_try[1:num + 1]).all()):
                alpha *= 0.5
                continue

            self.z[:] = z_try
            ss1 = float(self._eqns(self.rhs2))

            if np.isfinite(ss1) and (ss1 <= ss_best):
                ss_best = ss1
                z_best = z_try
                # accept immediately if improvement is adequate
                if ss1 <= ss0:
                    break

            alpha *= 0.5

        # Commit best found (or revert if none acceptable)
        self.z[:] = z_best

        corr = float(np.mean(np.abs((z_best[10:n + 11] - z0[10:n + 11]))))
        return corr


    def _compute_Y_and_B(self):
        """
        Port of the "slow Fourier transform" block in Fourier.cpp after convergence.
        Produces B[1..n] and Y[0..n] from final z[].
        """
        # Numba-accelerated kernel (drops back to pure NumPy if Numba is unavailable).
        if NUMBA_AVAILABLE:
            _compute_Y_and_B_numba(self.z, self.B, self.Y, self.cosa, self.n)
            return

        n = self.n
        z = self.z
        self.Y[:] = 0.0

        for j in range(1, n + 1):
            self.B[j] = z[j + n + 10]

            s = 0.5 * (z[10] + z[n + 10] * ((-1.0) ** j))
            for m in range(1, n):
                s += z[10 + m] * self.cosa[(m * j) % (2 * n)]
            self.Y[j] = 2.0 * s / n

    def _surface_keta(self, X):
        """
        Port of C++ Surface(double x): returns k(eta-d) at phase X (0..pi).
        """
        # Numba-accelerated kernel (drops back to pure NumPy if Numba is unavailable).
        if NUMBA_AVAILABLE:
            return float(_surface_keta_numba(self.Y, self.n, float(X)))

        n = self.n
        kEta = 0.0
        for j in range(1, n):
            kEta += self.Y[j] * np.cos(j * X)
        kEta += 0.5 * self.Y[n] * np.cos(n * X)
        return float(kEta)

    def _point(self, X, Y):
        """
        Port of the finite-depth branch of C++ Point(X,Y). Returns:
          u_dimless (w.r.t sqrt(g d))
          v_dimless (w.r.t sqrt(g d))
          dudt_dimless (w.r.t g)
        Input:
          X : phase in radians (k x)
          Y : vertical coordinate in wave scaling (k(z-d))
        """
        # Numba-accelerated kernel (drops back to pure NumPy if Numba is unavailable).
        if NUMBA_AVAILABLE:
            return _point_numba(float(X), float(Y), self.z, self.Tanh, self.B, self.n)

        n = self.n
        kd = float(self.z[1])

        # depth-scaled dimensionless bulk values
        c = float(self.z[4] / np.sqrt(kd))
        ce = float(self.z[5] / np.sqrt(kd))
        R = float(1.0 + self.z[9] / kd)

        # local variables in wave scaling
        u = 0.0
        v = 0.0
        ux = 0.0
        vx = 0.0

        for j in range(1, n + 1):
            Cos = np.cos(j * X)
            Sin = np.sin(j * X)

            coshdelta = np.cosh(j * Y)
            sinhdelta = np.sinh(j * Y)
            C = coshdelta + sinhdelta * self.Tanh[j]
            S = sinhdelta + coshdelta * self.Tanh[j]

            Bj = self.B[j]
            u += j * Bj * C * Cos
            v += j * Bj * S * Sin
            ux += -(j * j) * Bj * C * Sin
            vx += (j * j) * Bj * S * Cos

        # convert to depth scaling (see C++ comments)
        inv_kd_sqrt = 1.0 / np.sqrt(kd)
        inv_kd_32 = 1.0 / (kd ** 1.5)

        u *= inv_kd_sqrt
        v *= inv_kd_sqrt
        ux *= np.sqrt(kd)
        vx *= np.sqrt(kd)

        # add Euler current to u
        u = ce + u

        # time derivatives (steady in moving frame)
        ut = -c * ux
        vt = -c * vx
        uy = vx
        vy = -ux

        dudt = ut + u * ux + v * uy
        # dvdt is available if needed:
        # dvdt = vt + u * vx + v * vy

        # Bernoulli/pressure are not required by GUI, but kept for completeness:
        # y = 1.0 + Y / kd
        # Pressure = R - y - 0.5 * (((u - c) ** 2) + v * v)

        return float(u), float(v), float(dudt)

    # ----------------------------- public methods -----------------------------

    def solve(self):
        """
        Solve the steady nonlinear wave problem using Fenton's Fourier / stream-function method.

        The implementation follows the reference C++ structure:
        - Continuation in wave height (nstep)
        - Newton iterations with finite-difference Jacobian
        - Linear solve via SVD with Press-style truncation

        Robustness features:
        - Fail-fast on NaNs/Infs before calling LAPACK (prevents silent stalls).
        - Clear convergence state + message for GUI reporting.
        - Increased iteration budget automatically enabled for large |U|.
        """
        # Default outcome: failure unless we reach the end successfully
        self.converged = False
        self.last_error = ""

        # Basic input screening (physical requirements)
        if self.H_target <= 0.0 or self.T_target <= 0.0 or self.d <= 0.0:
            self.last_error = "Invalid inputs: H, T, and d must be > 0."
            return

        old_err = np.geterr()
        try:
            # Make numerical faults explicit. Underflow is benign in this context.
            np.seterr(over="raise", invalid="raise", divide="raise", under="ignore")

            # continuation step sizes
            dhe = self.Height / self.nstep
            dho = self.MaxH / self.nstep

            # height stepping
            for ns in range(1, self.nstep + 1):
                self.height = ns * dhe
                self.Hoverd = ns * dho

                # initial/extrapolated guess
                if ns == 1:
                    self._init_linear()
                else:
                    # z[i] = 2*sol[i][2] - sol[i][1]
                    self.z[1:self.num + 1] = (
                        2.0 * self.sol[1:self.num + 1, 2]
                        - self.sol[1:self.num + 1, 1]
                    )
                    # Fallback: if extrapolation produces an invalid start state,
                    # use the last converged state.
                    # This does not change any equations; it only prevents the
                    # Newton step from starting from NaN/Inf.
                    if (not np.isfinite(self.z[1:self.num + 1]).all()) or (self.z[1] <= 0.0):
                        self.z[1:self.num + 1] = self.sol[1:self.num + 1, 2]
                    if (not np.isfinite(self.z[1:self.num + 1]).all()) or (self.z[1] <= 0.0):
                        raise FloatingPointError(
                            "Invalid extrapolated start state for continuation step."
                        )

                # Newton iterations
                step_converged = False
                for it in range(1, self.number + 1):
                    # Newton iteration. If the start state was poisoned by extrapolation,
                    # retry once from the last converged state (no extrapolation).
                    try:
                        err = self._newton(it)
                    except FloatingPointError:
                        if (ns > 1) and (it == 1):
                            self.z[1:self.num + 1] = self.sol[1:self.num + 1, 2]
                            err = self._newton(it)
                        else:
                            raise

                    if not np.isfinite(err):
                        raise FloatingPointError("Non-finite Newton correction.")

                    # -----------------------------------------------------------------
                    # IMPORTANT: Update continuation storage BEFORE the convergence break.
                    # Otherwise, sol[:,2] may remain at a previous iterate, and the next
                    # continuation-step extrapolation can start from an invalid state.
                    # -----------------------------------------------------------------
                    if ns == 1:
                        self.sol[1:self.num + 1, 2] = self.z[1:self.num + 1]
                    else:
                        self.sol[1:self.num + 1, 1] = self.sol[1:self.num + 1, 2]
                        self.sol[1:self.num + 1, 2] = self.z[1:self.num + 1]

                    # Protect linear algebra calls on diverging states
                    if (not np.isfinite(self.z[1:self.num + 1]).all()) or (self.z[1] <= 0.0):
                        raise FloatingPointError(
                            "Divergence: non-finite/invalid state vector encountered."
                        )

                    criter = self.criter_final if (ns == self.nstep) else self.crit
                    if (it > 1) and (err < criter * abs(self.z[1])):
                        step_converged = True
                        break


                if not step_converged:
                    self.last_error = (
                        f"Newton did not converge within {self.number} iterations "
                        f"at continuation step {ns}/{self.nstep}."
                    )
                    return

                # update Y and B for this step (C++ does this each step)
                self._compute_Y_and_B()

            # ------------------------- dimensional post-process --------------------

            kd = float(self.z[1])
            if (not np.isfinite(kd)) or (kd <= 0.0):
                raise FloatingPointError("Invalid wavenumber (kd).")

            k_phys = kd / self.d
            L_phys = 2.0 * np.pi / k_phys
            c_dimless = float(self.z[4] / np.sqrt(kd))   # c / sqrt(g d)
            c_phys = c_dimless * np.sqrt(self.g * self.d)

            if (not np.isfinite(L_phys)) or (L_phys <= 0.0):
                raise FloatingPointError("Invalid wavelength.")
            if not np.isfinite(c_phys):
                raise FloatingPointError("Invalid celerity.")

            # surface nodes correspond to m*pi/n (half wave: crest->trough)
            eta_nodes = np.zeros(self.n + 1, dtype=DTYPE)
            for m in range(0, self.n + 1):
                kEta = float(self.z[10 + m])             # k(eta-d) at node
                eta_nodes[m] = self.d * (1.0 + kEta / kd)

            self.eta_nodes = eta_nodes
            self.k = float(k_phys)
            self.L = float(L_phys)
            self.c = float(c_phys)

            # store Bj as 0-based array for external use
            self.Bj = self.B[1:self.n + 1].copy()

            # crest/trough elevations relative to SWL
            self.eta_crest = float(self.eta_nodes[0] - self.d)
            self.eta_trough = float(self.eta_nodes[-1] - self.d)

            # non-dimensional descriptors
            self.steepness = self.H_target / self.L
            self.rel_depth = self.d / self.L
            self.ursell = (self.H_target * self.L * self.L) / (self.d ** 3)

            # regime classification (engineering convenience)
            if self.rel_depth < 0.05:
                self.regime = "Shallow"
            elif self.rel_depth < 0.5:
                self.regime = "Intermediate"
            else:
                self.regime = "Deep"

            # Miche breaking limit (keep GUI label/behaviour)
            self.breaking_limit_miche = float(0.142 * self.L * np.tanh(self.k * self.d))
            if self.breaking_limit_miche > 0:
                self.breaking_index = float(
                    self.H_target / self.breaking_limit_miche
                )
            else:
                self.breaking_index = 0.0
            self.is_breaking = bool(
                self.breaking_limit_miche > 0
                and self.H_target > self.breaking_limit_miche
            )

            # Integral properties (from C++ invariants, then dimensionalised)
            self._calc_integral_props_cpp()

            # Kinematics summary (crest/trough surface and bed under crest)
            self.u_bed, _, _ = self.get_kinematics(0.0, 0.0)

            # Quadratic bed shear estimate kept as-is (engineering heuristic)
            cf_est = 0.005
            self.tau_bed = 0.5 * RHO * cf_est * (self.u_bed ** 2)

            self.ExcursionBed = abs(self.u_bed) * self.T_target / (2.0 * np.pi)

            # crest/trough surface velocities
            self.u_surf, _, _ = self.get_kinematics(self.d + self.eta_crest, 0.0)
            u_trough, _, _ = self.get_kinematics(self.d + self.eta_trough, np.pi)
            self.asymmetry = abs(self.u_surf / u_trough) if abs(u_trough) > 0 else 0.0

            # scan phases for max vertical velocity and horizontal acceleration on the surface
            scan_phases = np.linspace(0.0, np.pi, 40)
            max_ax = 0.0
            max_w = 0.0
            for X in scan_phases:
                kEta = self._surface_keta(X)
                z_surf = self.d * (1.0 + kEta / kd)  # absolute from bed [m]
                _, w, ax = self.get_kinematics(z_surf, X)
                max_ax = max(max_ax, abs(ax))
                max_w = max(max_w, abs(w))

            self.acc_max = float(max_ax)
            self.w_max = float(max_w)

            self.converged = True

        except FloatingPointError as e:
            self.last_error = f"Floating point failure: {e}"
            self.converged = False
        except Exception as e:
            self.last_error = f"Solver error: {e}"
            self.converged = False
        finally:
            np.seterr(**old_err)

    def get_kinematics(self, z_bed, phase=0.0):
        """
        GUI-facing kinematics: (u_abs, w_abs, a_x) at a given vertical position.

        Parameters
        ----------
        z_bed : float
            Vertical coordinate from the bed [m]. Bed=0, mean level=d.
        phase : float
            Phase angle X = kx in radians (0 at crest, pi at trough for half-wave).

        Returns
        -------
        u_abs : float [m/s]
        w_abs : float [m/s]
        ax    : float [m/s^2]
        """
        kd = float(self.z[1])
        if kd <= 0.0 or self.d <= 0.0:
            return 0.0, 0.0, 0.0

        k_phys = kd / self.d
        X = float(phase)
        Y = float(k_phys * (float(z_bed) - self.d))  # wave scaling: k(z-d)

        u_nd, v_nd, dudt_nd = self._point(X, Y)

        u_abs = u_nd * np.sqrt(self.g * self.d)
        w_abs = v_nd * np.sqrt(self.g * self.d)
        ax = dudt_nd * self.g

        return float(u_abs), float(w_abs), float(ax)

    def _mean_square_bed_orbital_velocity(self, nph=720):
        """
        Mean square near-bed *orbital* horizontal velocity [m^2/s^2].

        Definition adopted (non-negative by construction):
            u_b^2 = < (u_b(t) - ū₁)^2 >
        where ū₁ is the Eulerian current (U), and <·> denotes averaging over
        one wave period (equivalently one wavelength for a steady progressive wave).

        Notes
        -----
        - This matches the standard RMS-orbital-velocity concept used in coastal
          engineering and sediment/force calculations: U_rms = sqrt(<u^2>).
        - Computed numerically by sampling phase uniformly.
        """
        if self.d <= 0.0 or self.T_target <= 0.0:
            return 0.0

        # Sample one full cycle in phase X = kx (0..2π). For a steady wave,
        # spatial averaging over one wavelength equals temporal averaging at a point.
        phases = np.linspace(0.0, 2.0 * np.pi, int(max(36, nph)), endpoint=False)

        ub2 = 0.0
        for ph in phases:
            u_abs, _, _ = self.get_kinematics(z_bed=0.0, phase=float(ph))  # bed: z_bed=0
            u_orb = u_abs - float(self.U)  # remove imposed Eulerian current
            ub2 += u_orb * u_orb

        return float(ub2 / len(phases))


    # ------------------------ integral properties (C++ parity) ----------------

    def _momentum_flux_S_depth(self, phase=0.0, npts=1200):
        """
        Compute the depth-scaled momentum flux S/(ρ g d²) in the *moving frame*:
            S = ∫₀^{η} [ p + ρ (u-c)² ] dz   (per unit crest width)

        This is the quantity printed by Fenton in Solution-Flat.res as:
            Momentum flux      S/(ρ g d²)

        Notes
        -----
        - Uses the same non-dimensional pressure from Bernoulli as in Point().
        - Evaluated at a single phase; the result is invariant with phase for the steady solution.
        """
        kd = float(self.z[1])
        if kd <= 0.0:
            return 0.0

        c = float(self.z[4] / np.sqrt(kd))          # c/√(g d)
        R = float(1.0 + self.z[9] / kd)            # R/(g d)

        X = float(phase)
        kEta = float(self._surface_keta(X))
        eta_over_d = 1.0 + kEta / kd               # y = z/d at free surface

        ys = np.linspace(0.0, eta_over_d, int(max(50, npts)), dtype=DTYPE)
        integ = np.zeros_like(ys)
        for idx, y in enumerate(ys):
            Y = kd * (y - 1.0)                     # Y = k(z-d) = kd*(y-1)
            u_nd, v_nd, _ = self._point(X, Y)      # u,v scaled by √(g d)
            urel = u_nd - c
            # Pressure scaled by (ρ g d) (same expression as C++/Point())
            P = R - y - 0.5 * (urel * urel + v_nd * v_nd)
            integ[idx] = P + urel * urel

        return float(_np_trapz(integ, ys))

    def _calc_integral_props_cpp(self):
        """
        Compute integral quantities using the same invariants as the C++ Output().
        Values are dimensionalised to match the GUI units.
        """
        kd = float(self.z[1])
        if kd <= 0.0:
            self.Power = self.EnergyDensity = self.Sxx = self.Impulse = self.Cg = 0.0
            self.MassTransport = 0.0
            self.BernoulliR = 0.0
            return

        # depth-scaled dimensionless bulk quantities
        c_dimless = float(self.z[4] / np.sqrt(kd))
        ce_dimless = float(self.z[5] / np.sqrt(kd))
        cs_dimless = float(self.z[6] / np.sqrt(kd))
        ubar_dimless = float(self.z[7] / np.sqrt(kd))

        Q_dimless = float(ubar_dimless - self.z[8] / (kd ** 1.5))
        R_dimless = float(1.0 + self.z[9] / kd)

        # C++ invariants in wave-number scaling
        pulse = float(self.z[8] + kd * self.z[5])
        ke = 0.5 * (self.z[4] * pulse - self.z[5] * Q_dimless * (kd ** 1.5))

        pe = 0.0
        for i in range(1, self.n + 1):
            pe += 0.25 * (self.Y[i] ** 2)

        ub2 = float(2.0 * self.z[9] - self.z[4] * self.z[4])
        q_term = float(self.z[7] * kd - self.z[8])

        sxx = float(4.0 * ke - 3.0 * pe + ub2 * kd + 2.0 * self.z[5] * q_term)
        f = float(
            self.z[4] * (3.0 * ke - 2.0 * pe)
            + 0.5 * ub2 * (pulse + self.z[4] * kd)
            + self.z[4] * self.z[5] * q_term
        )

        # Convert to depth-scaled dimensionless (as in C++ second column)
        E_depth = float((ke + pe) / (kd ** 2))
        KE_depth = float(ke / (kd ** 2))
        PE_depth = float(pe / (kd ** 2))

        # Store depth-scaled invariants (used for Solution-Flat style reporting)
        self.E_depth = E_depth
        self.KE_depth = KE_depth
        self.PE_depth = PE_depth
        Sxx_depth = float(sxx / (kd ** 2))
        F_depth = float(f / (kd ** 2.5))
        I_depth = float(pulse / (kd ** 1.5))

        self.Sxx_depth = Sxx_depth
        self.F_depth = F_depth
        self.I_depth = I_depth

        # Dimensionalise to GUI units
        self.EnergyDensity = float(RHO * self.g * (self.d ** 2) * E_depth)         # [J/m^2]
        self.Sxx = float(RHO * self.g * (self.d ** 2) * Sxx_depth)                 # [N/m]
        self.Power = float(RHO * (self.g ** 1.5) * (self.d ** 2.5) * F_depth)      # [W/m]
        # Momentum flux (Solution-Flat row 13) in moving frame
        self.MomentumFluxDepth = self._momentum_flux_S_depth(phase=0.0, npts=1200)  # S/(ρ g d²)
        self.MomentumFlux = float(
            RHO * self.g * (self.d ** 2) * self.MomentumFluxDepth
        )  # [N/m]
        self.Impulse = float(
            RHO * np.sqrt(self.g * (self.d ** 3)) * I_depth
        )  # [kg/(m·s)] per unit crest width

        self.BernoulliR = float(
            R_dimless * self.g * self.d
        )  # [m^2/s^2] head* g? (consistent scalar)
        self.MassTransport = float(
            cs_dimless * np.sqrt(self.g * self.d)
        )  # [m/s] (Stokes current)

        # Convenience values for Solution-Flat style reporting
        self.EulerianCurrent = float(self.U)                              # u1 [m/s]
        self.StokesCurrent = float(self.MassTransport)                      # u2 [m/s]
        self.MeanFluidSpeed = float(ubar_dimless * np.sqrt(self.g * self.d))  # Ū [m/s]
        self.VolumeFluxQ = float(Q_dimless * np.sqrt(self.g * (self.d ** 3))) # Q [m^2/s]
        self.WaveVolumeFlux_q = float(self.MeanFluidSpeed * self.d - self.VolumeFluxQ)  # q [m^2/s]
        self.BernoulliR_dimless = float(R_dimless)                          # R/(g d)
        self.Bernoulli_r = float(
            (R_dimless - 1.0) * self.g * self.d
        )  # r = R - g d [m^2/s^2]
        self.KineticEnergy = float(RHO * self.g * (self.d ** 2) * self.KE_depth)     # [J/m^2]
        self.PotentialEnergy = float(RHO * self.g * (self.d ** 2) * self.PE_depth)  # [J/m^2]
        # Mean square of bed orbital velocity (Solution-Flat row 17):
        # u_b^2 = < (u_b(t) - ū₁)^2 >  (RMS-orbital-velocity definition; non-negative)
        self.MeanSquareBedVelocity = float(
            self._mean_square_bed_orbital_velocity(nph=720)
        )  # [m^2/s^2]

        if abs(self.EnergyDensity) > 1e-12:
            self.Cg = float(self.Power / self.EnergyDensity)
        else:
            self.Cg = 0.0




def L_wave(H: float, T: float, d: float, U: float = 0.0) -> float:
    """
    Compute wavelength L [m] for a steady periodic wave using Fenton's stream-function solver.

    Parameters
    ----------
    H  : wave height [m]
    T  : wave period [s]
    d  : water depth [m]
    U  : Eulerian current, positive with wave propagation [m/s]

    Returns
    -------
    L : wavelength [m]

    Raises
    ------
    RuntimeError if the underlying nonlinear solve does not converge.
    """
    solver = FentonStreamFunction(H=H, T=T, d=d, U=U)
    solver.solve()
    if not solver.converged:
        raise RuntimeError(solver.last_error or "Fenton solver did not converge.")
    return float(solver.L)


def L(H: float, T: float, d: float, U: float = 0.0) -> float:
    """Alias for L_wave(...) to match expected external API (fenton.L)."""
    return L_wave(H, T, d, U)


def _prompt_float(prompt: str, default: float) -> float:
    s = input(f"{prompt} [{default}]: ").strip()
    if not s:
        return float(default)
    return float(s)


def _cli(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="function.py",
        description=(
            "Fenton nonlinear wavelength calculator (finite depth, period input, "
            "Eulerian current)."
        ),
    )
    parser.add_argument("H", nargs="?", type=float, help="Wave height H [m]")
    parser.add_argument("T", nargs="?", type=float, help="Wave period T [s]")
    parser.add_argument("d", nargs="?", type=float, help="Water depth d [m]")
    parser.add_argument("U", nargs="?", type=float, help="Eulerian current U [m/s]")
    args = parser.parse_args(argv)

    # No args -> interactive prompts (defaults: H=3, T=5, d=9, U=1)
    if args.H is None and args.T is None and args.d is None and args.U is None:
        H = _prompt_float("Wave height H (m)", 3.0)
        T = _prompt_float("Wave period T (s)", 9.0)
        d = _prompt_float("Water depth d (m)", 5.0)
        U = _prompt_float("Eulerian current U (m/s)", 1.0)
    else:
        if None in (args.H, args.T, args.d, args.U):
            parser.error(
                "Provide all four arguments: H T d U, or run without arguments "
                "for interactive mode."
            )
        H, T, d, U = float(args.H), float(args.T), float(args.d), float(args.U)

    Lval = L(H, T, d, U)
    print(f"L = {Lval:.12g} m")
    return 0


__all__ = ["L", "L_wave", "FentonStreamFunction"]


if __name__ == "__main__":
    raise SystemExit(_cli())
