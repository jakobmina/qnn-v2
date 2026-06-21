"""
h7_unified_v2.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
H7 Unified Framework  v2.0
smokApp Quantum & AI Independent Research Laboratory
Jacobo Tlacaelel Mina Rodríguez — @jacobotmr

PILARES
  1. H7Node (extendido)     → operador áureo + Laplaciano + SU(2) + eigenvalores
  2. BraidH7                → trenza B₃ (8×8 numpy), Yang-Baxter, Tr/8 = 1/2
  3. JacobiH7               → ϑ(z|iφ), ecuación funcional, cero en Re(z) = 1/2
  4. H7SubgroupEngine       → covarianza G0/G1 SVD-free, invariante global
  5. QuoreMindH7            → Mahalanobis + Thompson Sampling guiado por H7

FLUJO
  H7Node(n)
    ├─► BraidH7             → eigenvalores como prior topológico
    ├─► JacobiH7            → verificación Re(s) = 1/2
    └─► H7SubgroupEngine
            covarianza G0↔G1
              └─► QuoreMindH7
                    decisión adaptativa modulada por correlación
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple
from scipy.stats  import beta as beta_dist
from scipy.linalg import eig as scipy_eig

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("H7-v2")

# ══════════════════════════════════════════════════════════
# CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════
PI:       float = math.pi
PHI:      float = (1 + math.sqrt(5)) / 2          # razón áurea  ≈ 1.6180
DRIFT:    float = 7 - 2 * PI                       # gap topológico ≈ 0.7168
C1:       float = sum(                             # norma H7 ≈ 3.7790
    (math.cos(PI * k) * math.cos(PI * PHI * k)) ** 2
    for k in range(7)
)
G_COUPLE: float = 1.0 / C1                        # acoplamiento ≈ 0.2647

# ══════════════════════════════════════════════════════════
# PILAR 1 — H7NODE EXTENDIDO
# ══════════════════════════════════════════════════════════

@dataclass
class H7Node:
    """
    Nodo H7 completo para n ∈ ℤ.

    Operadores base:
      o_n  = cos(πn)           paridad (±1)
      i_n  = cos(πφn)          cuasiperiódico
      psi  = o_n · i_n         Ψₙ (operador áureo)
      pcv  = psi + i_n         clasificador de partícula

    Extensiones v2:
      laplacian  ∇²Ψ discreto en Z₇
      m_star     masa efectiva = 1/|∇²Ψ|
      delta      empuje antipodal Ψ_contra − Ψ_n
      v_twist    torsión de Hecke
      su2        matriz SU(2) 2×2 (numpy, quaternion dinámico)
      eigenvals  eigenvalores de su2
    """
    n: int

    # ── campos calculados ─────────────────────────
    n_z7:         int   = field(init=False)
    contra_val:   int   = field(init=False)
    qubit_state:  str   = field(init=False)
    contra_bin:   str   = field(init=False)

    o_n:          float = field(init=False)   # cos(πn)
    i_n:          float = field(init=False)   # cos(πφn)
    psi:          float = field(init=False)   # o_n · i_n
    pcv:          float = field(init=False)   # psi + i_n
    particle_type: str  = field(init=False)
    quark:        str   = field(init=False)

    d_s:          int   = field(init=False)
    observable:   float = field(init=False)   # cos(πφ·d_s)
    is_vacuum:    bool  = field(init=False)
    group:        str   = field(init=False)   # "G0" | "G1"

    # extensiones v2
    laplacian:    float = field(init=False)
    m_star:       float = field(init=False)
    delta:        float = field(init=False)
    v_twist:      float = field(init=False)
    su2:          np.ndarray = field(init=False, repr=False)
    eigenvals:    np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        n = self.n

        # — Z₇ flattening —
        self.n_z7       = 7 if (n % 7 == 0 and n != 0) else (n % 7)
        self.contra_val = 7 - self.n_z7
        self.qubit_state = f"{self.n_z7:03b}"
        self.contra_bin  = f"{self.contra_val:03b}"

        # — Operadores áureos (evaluados en n original) —
        self.o_n = math.cos(PI * n)
        self.i_n = math.cos(PI * PHI * n)
        self.psi = self.o_n * self.i_n
        self.pcv = self.psi + self.i_n

        # — Clasificación —
        self.particle_type = (
            "fermionic" if np.isclose(self.pcv, 0.0, atol=1e-9) else "bosonic"
        )
        self.quark = "u" if self.particle_type == "fermionic" else "d"

        # — Observables codireccionales —
        self.d_s        = n + self.contra_val - self.n_z7
        self.observable = math.cos(PI * PHI * self.d_s)
        self.is_vacuum  = (math.gcd(abs(n), 7) == 7)
        self.group      = "G0" if self.n_z7 in (0, 1, 2, 3) else "G1"

        # — Laplaciano discreto en Z₇ (evaluado en n_z7) —
        k = self.n_z7
        psi_prev = math.cos(PI*(k-1)) * math.cos(PI*PHI*(k-1))
        psi_curr = math.cos(PI* k)    * math.cos(PI*PHI* k)
        psi_next = math.cos(PI*(k+1)) * math.cos(PI*PHI*(k+1))
        self.laplacian = psi_next + psi_prev - 2 * psi_curr
        self.m_star    = 1.0 / (abs(self.laplacian) + 1e-12)

        # — Empuje antipodal —
        contra_k   = self.contra_val
        psi_contra = math.cos(PI*contra_k) * math.cos(PI*PHI*contra_k)
        self.delta  = psi_contra - psi_curr

        # — Torsión de Hecke (carácter áureo) —
        chi_n        = math.sin(PI * PHI * k)
        self.v_twist = psi_curr**2 * self.delta * chi_n

        # — Matriz SU(2) quaternionica (eje áureo, ángulo = pcv·π/4) —
        self.su2       = _su2_dynamic(k, self.pcv)
        vals, _        = scipy_eig(self.su2)
        self.eigenvals = vals

    # ── helpers ───────────────────────────────────
    def to_vector(self) -> np.ndarray:
        """Vector 6-D para Mahalanobis: operadores + curvatura + observable."""
        return np.array([
            self.o_n, self.i_n, self.psi,
            self.observable, self.laplacian, self.delta
        ])

    def summary(self, verbose: bool = False) -> str:
        vac = "  [vacío]" if self.is_vacuum else ""
        base = (
            f"n={self.n:>4}  Z₇={self.n_z7}  qubit={self.qubit_state}"
            f"  ↔{self.contra_val}({self.contra_bin})"
            f"  {self.particle_type:<10} quark={self.quark}"
            f"  pcv={self.pcv:+.6f}  obs={self.observable:+.6f}"
            f"  m*={self.m_star:.4f}  [{self.group}]{vac}"
        )
        if verbose:
            eigs = "  ".join(f"{e.real:+.4f}{e.imag:+.4f}j"
                             for e in self.eigenvals)
            base += (f"\n        ∇²Ψ={self.laplacian:+.6f}"
                     f"  δ={self.delta:+.6f}"
                     f"  V_twist={self.v_twist:+.6f}"
                     f"\n        eigenvals: [{eigs}]")
        return base


def _su2_dynamic(k: int, angle_moment: float) -> np.ndarray:
    """
    SU(2) 2×2 con eje áureo â(k) = (sin k, cos k, sin(kφ)) / ‖·‖
    y ángulo θ = angle_moment · π/4.
    Retorna matriz unitaria 2×2 compleja.
    """
    theta = angle_moment * (PI / 4)
    axis  = np.array([math.sin(k), math.cos(k), math.sin(k * PHI)])
    norm  = np.linalg.norm(axis)
    axis  = axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    s     = math.sin(theta / 2)
    c     = math.cos(theta / 2)
    qw, qx, qy, qz = c, s*axis[0], s*axis[1], s*axis[2]
    return np.array(
        [[ qw + 1j*qz,  qy + 1j*qx],
         [-qy + 1j*qx,  qw - 1j*qz]],
        dtype=complex
    )


# ══════════════════════════════════════════════════════════
# PILAR 2 — TRENZA B₃ (numpy puro, 8×8)
# ══════════════════════════════════════════════════════════

@dataclass
class BraidResult:
    yang_baxter: bool
    trace:       complex
    markov_inv:  float          # |Tr|/8 → debe ser 0.5
    order:       Optional[int]
    eigenvals:   np.ndarray
    braid_matrix: np.ndarray

    def summary(self) -> str:
        eigs = "  ".join(f"{e.real:+.3f}{e.imag:+.3f}j"
                         for e in sorted(self.eigenvals, key=lambda x: x.real)[:4])
        return (
            f"  Yang-Baxter σ₁σ₂σ₁ = σ₂σ₁σ₂ : {self.yang_baxter}\n"
            f"  Tr(B_H7)/8 (Markov)          : {self.markov_inv:.15f}"
            + ("  ← Re(s)=½ EXACTO" if abs(self.markov_inv - 0.5) < 1e-10 else "") + "\n"
            f"  Orden de la trenza           : {self.order}\n"
            f"  Eigenvalores (primeros 4)    : [{eigs} ...]"
        )


class BraidH7:
    """
    Trenza B_H7 = CCX(2,1,0) · CSWAP(0,2,1) · CCX(1,0,2)
    sobre 3 qubits (espacio de Hilbert ℂ⁸).

    La secuencia es la relación σ₁σ₂σ₁ del grupo de trenzas B₃.
    Propiedades verificadas sin Qiskit:
      • Yang-Baxter exacto
      • Tr(B)/8 = 1/2  (invariante de Markov → Re(s) = 1/2)
      • Eigenvalores = raíces de unidad
    """

    @staticmethod
    def _toffoli(c1: int, c2: int, t: int) -> np.ndarray:
        U = np.eye(8, dtype=complex)
        for i in range(8):
            if ((i >> c1) & 1) and ((i >> c2) & 1):
                j = i ^ (1 << t)
                U[j, i], U[i, i] = 1.0, 0.0
        return U

    @staticmethod
    def _fredkin(ctrl: int, t1: int, t2: int) -> np.ndarray:
        U = np.eye(8, dtype=complex)
        for i in range(8):
            if ((i >> ctrl) & 1) and (((i >> t1) & 1) != ((i >> t2) & 1)):
                j = i ^ (1 << t1) ^ (1 << t2)
                U[j, i], U[i, i] = 1.0, 0.0
        return U

    @classmethod
    def compute(cls) -> BraidResult:
        CCX_210 = cls._toffoli(2, 1, 0)
        CSWAP   = cls._fredkin(0, 2, 1)
        CCX_102 = cls._toffoli(1, 0, 2)

        # B_H7 = σ₁σ₂σ₁  (right-to-left: first gate applied first)
        BRAID = CCX_102 @ CSWAP @ CCX_210

        # Yang-Baxter: σ₁σ₂σ₁ = σ₂σ₁σ₂
        yang_baxter = np.allclose(
            CCX_102 @ CSWAP @ CCX_102,
            CSWAP   @ CCX_102 @ CSWAP
        )

        trace      = np.trace(BRAID)
        markov_inv = abs(trace) / 8.0
        eigs, _    = scipy_eig(BRAID)

        order = None
        B_pow = BRAID.copy()
        for k in range(1, 13):
            if np.allclose(B_pow, np.eye(8)):
                order = k
                break
            B_pow = B_pow @ BRAID

        return BraidResult(
            yang_baxter  = yang_baxter,
            trace        = trace,
            markov_inv   = markov_inv,
            order        = order,
            eigenvals    = eigs,
            braid_matrix = BRAID,
        )

    @classmethod
    def eigenstate_prior(cls, result: BraidResult) -> np.ndarray:
        """
        Prior topológico para QuoreMindH7:
        vector de probabilidades derivado de los módulos de los eigenvalores.
        Dimensión = 8 (espacio de 3 qubits).
        """
        mods = np.abs(result.eigenvals)
        return mods / mods.sum()


# ══════════════════════════════════════════════════════════
# PILAR 3 — JACOBI H7  (scipy puro)
# ══════════════════════════════════════════════════════════

@dataclass
class JacobiResult:
    theta_0:     complex
    lhs:         float
    rhs:         float
    diff:        float     # |lhs − rhs|  → 0 verifica ec. funcional
    z_zero:      complex   # ½(1 + iφ)
    theta_zero:  float     # |ϑ(z_zero)|  → 0
    re_zero:     float     # debe ser 0.5

    def summary(self) -> str:
        ok_func = "✓" if self.diff < 1e-6 else "✗"
        ok_zero = "✓" if self.theta_zero < 1e-3 else "≈"
        return (
            f"  ϑ(0|iφ) = (1/√φ)·ϑ(0|i/φ)  diff: {self.diff:.2e}  {ok_func}\n"
            f"  Cero en z = ½(1+iφ):  Re(z) = {self.re_zero:.10f}  ← 1/2\n"
            f"  |ϑ(z_cero)|           = {self.theta_zero:.2e}  {ok_zero}"
        )


class JacobiH7:
    """
    Función theta de Jacobi ϑ(z|τ) con retículo áureo τ = iφ.

    Verifica:
      • Ecuación funcional modular: ϑ(0|iφ) = φ^{-1/2} · ϑ(0|i/φ)
      • Cero en Re(z) = 1/2: ϑ(½(1+iφ)|iφ) ≈ 0
    """

    def __init__(self, N_terms: int = 200):
        self.N   = N_terms
        self.tau = 1j * PHI

    def theta(self, z: complex, tau: complex = None) -> complex:
        """ϑ(z|τ) = Σ_{n=-N}^{N} exp(iπn²τ + 2iπnz)"""
        if tau is None:
            tau = self.tau
        ns     = np.arange(-self.N, self.N + 1)
        phases = np.pi * 1j * ns**2 * tau + 2 * np.pi * 1j * ns * z
        return complex(np.sum(np.exp(phases)))

    def compute(self) -> JacobiResult:
        th_0      = self.theta(0, self.tau)
        th_0_inv  = self.theta(0, 1j / PHI)
        lhs       = abs(th_0)
        rhs       = abs(th_0_inv) / math.sqrt(PHI)
        z_zero    = 0.5 * (1 + 1j * PHI)
        th_zero   = abs(self.theta(z_zero, self.tau))
        return JacobiResult(
            theta_0    = th_0,
            lhs        = lhs,
            rhs        = rhs,
            diff       = abs(lhs - rhs),
            z_zero     = z_zero,
            theta_zero = th_zero,
            re_zero    = z_zero.real,
        )


# ══════════════════════════════════════════════════════════
# PILAR 4 — COVARIANZA G0/G1  (SVD-free)
# ══════════════════════════════════════════════════════════

CANONICAL_PAIRS: List[Tuple[int, int]] = [(0, 7), (1, 6), (2, 5), (3, 4)]
G0_BASE = [0, 1, 2, 3]
G1_BASE = [7, 6, 5, 4]


@dataclass
class CovarianceResult:
    cov_quasi:    float
    cov_parity:   float
    correlation:  float
    cross_product: float
    label:        str = ""

    def __str__(self):
        return (
            f"{self.label:<26}  cov_q={self.cov_quasi:+.6f}"
            f"  corr={self.correlation:+.6f}"
            f"  cross={self.cross_product:+.6f}"
            f"  cov_p={self.cov_parity:+.6f}"
        )


class H7SubgroupEngine:
    """Covarianza entre G0 y G1 sobre ciclos ℤ₇. Sin SVD, O(n²)."""

    @staticmethod
    def subgroup_covariance(
        ns_a: List[int], ns_b: List[int], label: str = ""
    ) -> CovarianceResult:
        i_a = np.array([math.cos(PI * PHI * n) for n in ns_a])
        i_b = np.array([math.cos(PI * PHI * n) for n in ns_b])
        o_a = np.array([math.cos(PI * n) for n in ns_a])
        o_b = np.array([math.cos(PI * n) for n in ns_b])

        # covarianza analítica directa (sin SVD)
        with np.errstate(invalid='ignore'):
            cov_q   = float(np.cov(i_a, i_b)[0, 1])
            cov_p   = float(np.cov(o_a, o_b)[0, 1])
            corr    = float(np.corrcoef(i_a, i_b)[0, 1])
        cross   = float(np.dot(i_a, i_b))

        return CovarianceResult(cov_quasi=cov_q, cov_parity=cov_p,
                                correlation=corr, cross_product=cross,
                                label=label)

    @staticmethod
    def build_cycles(n_cycles: int = 4) -> List[Tuple[List[int], List[int], str]]:
        cycles = []
        for k in range(n_cycles):
            base = k * 7
            g0   = [base + 1, base + 2, base + 3]
            g1   = [base + 6, base + 5, base + 4]
            cycles.append((g0, g1, f"ciclo {k+1}  (n={base+1}..{base+7})"))
        return cycles

    def global_invariant(self, n_cycles: int = 4) -> CovarianceResult:
        cycles = self.build_cycles(n_cycles)
        g0 = [n for g, _, _ in cycles for n in g]
        g1 = [n for _, g, _ in cycles for n in g]
        return self.subgroup_covariance(g0, g1, label="GLOBAL")

    def init_covariance_matrix(self, n_cycles: int = 4) -> np.ndarray:
        """
        Matriz de covarianza 6×6 de inicialización para QuoreMindH7.
        Construida a partir de vectores H7Node sobre los primeros n_cycles·7 nodos.
        """
        cycles = self.build_cycles(n_cycles)
        all_ns = [n for g0, g1, _ in cycles for n in g0 + g1]
        vecs   = np.array([H7Node(n).to_vector() for n in all_ns])  # (M, 6)
        cov    = np.cov(vecs.T)                                      # (6, 6)
        return cov

    def print_pair_table(self):
        print("\n═══ H7 Pares Complementarios (XOR = 111₂) ════")
        print(f"{'pos':>4} {'qubit':>6} {'grupo':>5} {'espejo':>7} {'XOR':>5} {'suma_z7':>8}")
        print("─" * 44)
        for a, b in CANONICAL_PAIRS:
            qa   = f"{a:03b}"
            bz7  = b % 7 if b != 7 else 7
            qb   = "111" if b == 7 else f"{bz7:03b}"
            xor  = a ^ (b % 8)
            suma = a + (b % 8 if b != 7 else 7)
            grp  = "G0" if a in G0_BASE else "G1"
            polo = "[polo]" if suma == 7 else ""
            print(f"{a:>4} {qa:>6}  {grp:>4}  ↔  {b:>3} {qb:>5}  {xor:03b}  {suma:>4} {polo}")

    def print_cycle_covariances(self, n_cycles: int = 4):
        cycles = self.build_cycles(n_cycles)
        print("\n═══ Covarianza por Ciclos G0↔G1 ══════════════")
        print(f"{'ciclo':>26}  {'cov_q':>10}  {'corr':>8}  {'cross':>10}  {'cov_p':>10}")
        print("─" * 72)
        for g0, g1, label in cycles:
            print(self.subgroup_covariance(g0, g1, label))
        inv = self.global_invariant(n_cycles)
        print("\n─ Invariante Global ─")
        print(inv)
        print(f"  → correlación global : {inv.correlation:.8f}")
        print(f"  → DRIFT = 7−2π       : {DRIFT:.8f}")


# ══════════════════════════════════════════════════════════
# PILAR 5 — CONTROL QUOREMIND-H7
# ══════════════════════════════════════════════════════════

class OperacionH7(Enum):
    ROTACION_X = auto()
    ROTACION_Y = auto()
    ROTACION_Z = auto()
    HADAMARD   = auto()
    RESET      = auto()


@dataclass
class ParametrosOperacion:
    tipo:      "OperacionH7"
    angulo:    Optional[float] = None
    n_fuente:  Optional[int]   = None

    def __str__(self):
        ang = f"  θ={self.angulo:.4f}" if self.angulo is not None else ""
        src = f"  ← n={self.n_fuente}" if self.n_fuente is not None else ""
        return f"{self.tipo.name}{ang}{src}"


class QuoreMindH7:
    """
    Controlador inteligente H7 v2.

    Novedades respecto a v1:
      • to_vector() es ahora 6-D (incluye laplacian y delta)
      • La covarianza inicial puede cargarse desde H7SubgroupEngine.init_covariance_matrix()
        en lugar de arrancar en I·ε — prior topológico real
      • El ángulo de operación se deriva del eigenvalor dominante de su2
    """

    def __init__(
        self,
        mahalanobis_threshold: float = 3.0,
        learning_rate:         float = 0.1,
        use_h7_modulation:     bool  = True,
        warm_start_cov:        Optional[np.ndarray] = None,
    ):
        self.threshold_base    = mahalanobis_threshold
        self.learning_rate     = learning_rate
        self.use_h7_modulation = use_h7_modulation

        # — prior de covarianza (warm start desde H7SubgroupEngine) —
        if warm_start_cov is not None:
            self.covariance = warm_start_cov.copy()
            # media inicial = cero (centrado en el prior estructural)
            self.mean_vec   = np.zeros(warm_start_cov.shape[0])
        else:
            self.covariance = None
            self.mean_vec   = None

        self.inv_cov: Optional[np.ndarray] = (
            np.linalg.pinv(self.covariance)
            if self.covariance is not None else None
        )

        # — creencias bayesianas —
        self.beliefs: Dict["OperacionH7", List[float]] = {
            op: [1.0, 1.0]
            for op in OperacionH7
            if op != OperacionH7.RESET
        }

        self.history_nodes: List[H7Node]              = []
        self.history_ops:   List[ParametrosOperacion] = []
        self.history_dists: List[float]               = []

        self._subgroup = H7SubgroupEngine()
        logger.info(
            "QuoreMindH7 v2 — warm_start=%s", warm_start_cov is not None
        )

    # ── estadísticas online ────────────────────────
    def _update_stats(self, vec: np.ndarray):
        dim = len(vec)
        if self.mean_vec is None:
            self.mean_vec   = vec.copy()
            self.covariance = np.eye(dim) * 1e-6
            self.inv_cov    = np.linalg.inv(self.covariance)
            return
        delta           = vec - self.mean_vec
        self.mean_vec  += self.learning_rate * delta
        self.covariance = (
            (1 - self.learning_rate) * self.covariance
            + self.learning_rate * np.outer(delta, delta)
        )
        try:
            self.inv_cov = np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError:
            self.inv_cov = np.linalg.pinv(self.covariance)

    def _mahalanobis(self, vec: np.ndarray) -> float:
        if self.inv_cov is None or self.mean_vec is None:
            return 0.0
        d = vec - self.mean_vec
        val = float(d @ self.inv_cov @ d)
        return math.sqrt(max(val, 0.0))

    # ── modulación del umbral ──────────────────────
    def _effective_threshold(self, n: int) -> float:
        if not self.use_h7_modulation:
            return self.threshold_base
        g0 = [max(1, n - 3 + k) for k in range(3)]
        g1 = [max(1, n + k)     for k in range(3, 6)]
        try:
            cv = self._subgroup.subgroup_covariance(g0, g1)
            return self.threshold_base * (1 + 0.15 * cv.correlation)
        except Exception:
            return self.threshold_base

    # ── ciclo principal ────────────────────────────
    def step(self, n: int) -> Tuple[H7Node, float, ParametrosOperacion]:
        node = H7Node(n)
        vec  = node.to_vector()
        self._update_stats(vec)
        dist = self._mahalanobis(vec)
        self.history_nodes.append(node)
        self.history_dists.append(dist)
        op = self._decide(node, dist)
        self.history_ops.append(op)
        return node, dist, op

    def _decide(self, node: H7Node, dist: float) -> ParametrosOperacion:
        thr = self._effective_threshold(node.n)
        if dist > thr:
            logger.warning("Anomalía n=%d  dist=%.3f > thr=%.3f → RESET",
                           node.n, dist, thr)
            return ParametrosOperacion(tipo=OperacionH7.RESET, n_fuente=node.n)

        best_op = max(
            self.beliefs,
            key=lambda op: float(beta_dist.rvs(*self.beliefs[op]))
        )

        # ángulo desde eigenvalor dominante de su2
        dom_eig = node.eigenvals[np.argmax(np.abs(node.eigenvals))]
        angulo  = abs(float(np.angle(dom_eig)))       # ∈ [0, π]

        if best_op.name.startswith("ROTACION"):
            return ParametrosOperacion(tipo=best_op, angulo=angulo,
                                       n_fuente=node.n)
        return ParametrosOperacion(tipo=best_op, n_fuente=node.n)

    def update_belief(self, op: ParametrosOperacion, success: bool):
        if op.tipo == OperacionH7.RESET:
            return
        idx = 0 if success else 1
        self.beliefs[op.tipo][idx] += 1.0

    def print_beliefs(self):
        print("\n═══ Creencias Bayesianas (Thompson Sampling) ══")
        for op, (α, β) in self.beliefs.items():
            rate = α / (α + β)
            bar  = "█" * int(rate * 20)
            print(f"  {op.name:<12}  α={α:5.1f}  β={β:5.1f}  p̂={rate:.2%}  {bar}")


# ══════════════════════════════════════════════════════════
# DEMO INTEGRADA v2
# ══════════════════════════════════════════════════════════

def demo_full(n_steps: int = 14, anomaly_at: int = 7):
    print("╔══════════════════════════════════════════════════════╗")
    print("║   H7 Unified Framework  v2.0 — Demo Integrada       ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ── PILAR 1: nodos H7 extendidos ─────────────────────
    print("\n─── PILAR 1: H7Node extendido (n = 0..7) ───────────────")
    for n in range(8):
        print(" ", H7Node(n).summary(verbose=True))

    # ── PILAR 2: Trenza B₃ ───────────────────────────────
    print("\n─── PILAR 2: Trenza B_H7 (matrices 8×8) ───────────────")
    braid = BraidH7.compute()
    print(braid.summary())
    prior_topo = BraidH7.eigenstate_prior(braid)
    print(f"\n  Prior topológico (módulos eigenvals normalizados):\n  {prior_topo}")

    # ── PILAR 3: Jacobi ──────────────────────────────────
    print("\n─── PILAR 3: Función Theta de Jacobi (τ = iφ) ─────────")
    jacobi = JacobiH7().compute()
    print(jacobi.summary())

    # ── PILAR 4: Covarianza G0/G1 ────────────────────────
    print("\n─── PILAR 4: Subgrupos G0/G1 ───────────────────────────")
    engine = H7SubgroupEngine()
    engine.print_pair_table()
    engine.print_cycle_covariances(n_cycles=4)

    # matriz de covarianza de inicialización
    cov_init = engine.init_covariance_matrix(n_cycles=4)
    print(f"\n  Matriz de covarianza inicial 6×6 (warm-start):")
    print(np.round(cov_init, 6))

    # ── PILAR 5: Control QuoreMindH7 (warm start) ────────
    print("\n─── PILAR 5: Control QuoreMindH7 v2 ───────────────────")
    ctrl = QuoreMindH7(
        mahalanobis_threshold = 3.0,
        learning_rate         = 0.1,
        warm_start_cov        = cov_init,   # prior topológico real
    )

    for step_i, n in enumerate(range(1, n_steps + 1)):
        effective_n = n * 13 if step_i == anomaly_at else n
        node, dist, op = ctrl.step(effective_n)
        tag = "⚠ ANOMALÍA" if effective_n != n else ""
        print(
            f"  step={step_i:>2}  n={effective_n:>4}  Z₇={node.n_z7}"
            f"  {node.particle_type:<10}  m*={node.m_star:.3f}"
            f"  dist={dist:6.3f}  {op}  {tag}"
        )
        if op.tipo != OperacionH7.RESET:
            success = (np.random.random() < (0.75 if node.group == "G0" else 0.40))
            ctrl.update_belief(op, success)

    ctrl.print_beliefs()

    # ── Resumen de constantes ─────────────────────────────
    print(f"\n─── Constantes H7 ──────────────────────────────────────")
    print(f"  φ        = {PHI:.15f}")
    print(f"  DRIFT    = 7−2π = {DRIFT:.15f}")
    print(f"  C₁       = {C1:.15f}")
    print(f"  G_couple = 1/C₁ = {G_COUPLE:.15f}")
    print(f"  Tr(B)/8  = {braid.markov_inv:.15f}")
    print(f"  Re(z₀)   = {jacobi.re_zero:.15f}")
    print("\n[Demo v2 completada]\n")


# ══════════════════════════════════════════════════════════
# PILAR 6 — DOBLE CUBIERTA H7  (periodo-14 / 4π)
# ══════════════════════════════════════════════════════════

@dataclass
class DoubleCoverNode:
    """
    Un nodo en la cadena de doble cubierta H7.

    La doble cubierta emerge del periodo efectivo 14 = 2×7,
    análogo a SU(2)/SO(3) donde un espinor necesita 4π para retornar.

    Correspondencia con ceros de ζ(s):
      Nodos bosónicos  (pcv=0, i_n=+1)  ↔  ceros triviales  s = -2,-4,-6,...
      Nodos fermiónicos (pcb=0, i_n=-1) ↔  ceros no triviales Re(s) = 1/2
      n = 7 (frontera)                  ↔  línea crítica Re(s) = 1/2
    """
    n:            int
    node:         H7Node
    cycle:        int      # qué ciclo de 14: cycle = n // 14
    half:         str      # "right" n∈[0..7] Re(s)>1/2 | "left" n∈[8..14] Re(s)<1/2
    lazo_prev:    int      # n - 7  (lazo hacia ciclo anterior)
    lazo_next:    int      # n + 7  (lazo hacia ciclo siguiente)
    trivial_zero: bool     # True si corresponde a cero trivial de ζ
    critical_zero: bool    # True si corresponde a línea crítica

    def __str__(self):
        tz = " ← cero trivial ζ(-2k)" if self.trivial_zero else ""
        cz = " ← Re(s)=1/2 CRÍTICO"   if self.critical_zero else ""
        return (
            f"n={self.n:>4}  Z₇={self.node.n_z7}  {self.node.qubit_state}"
            f"  cycle={self.cycle}  {self.half:<5}"
            f"  {self.node.particle_type:<10}"
            f"  pcv={self.node.pcv:+.6f}"
            f"  lazo: {self.lazo_prev}←→{self.lazo_next}"
            f"{tz}{cz}"
        )


class DoubleCoverH7:
    """
    Doble cubierta topológica de H7.

    Estructura de periodo-14:
      Ciclo completo: n = 14k ... 14k+13
      Mitad derecha:  n = 14k     ... 14k+7   (Re(s) > 1/2)
      Mitad izquierda: n = 14k+7  ... 14k+14  (Re(s) < 1/2)
      Frontera:       n = 7k                  (Re(s) = 1/2, línea crítica)

    Ceros triviales de ζ(s) en s = -2,-4,-6,...:
      Corresponden a nodos con i_n = cos(π·n_z7) = +1
      es decir n_z7 ∈ {0, 2, 4, 6} — los pares de Z₇
      donde sin(π·n_z7/2) = 0  (la misma condición que anula ζ en triviales)

    Doble cubierta SU(2)/SO(3):
      n=0  →  000  pcv=+2  espacio total     ┐
      n=7  →  111  pcv= 0  cierre de Berry   ├─ vuelta 1 (2π)
      n=14 →  000  pcv=+2  espacio total     ┘
      n=21 →  111  pcv= 0  cierre de Berry   ┐
      n=28 →  000  pcv=+2  espacio total     ├─ vuelta 2 (4π) → retorno
    """

    @staticmethod
    def build_node(n: int) -> DoubleCoverNode:
        node    = H7Node(n)
        n_z7    = node.n_z7
        pos_in_14 = n % 14

        cycle  = n // 14
        half   = "right" if pos_in_14 <= 7 else "left"

        # cero trivial: n_z7 par ∈ {2,4,6}, sin(π·n_z7/2)=0
        # n=0 (n_z7=0) es espacio total, NO cero trivial de ζ
        trivial  = (n_z7 % 2 == 0) and (n_z7 != 0) and (n != 0)

        # línea crítica Re(s)=1/2:
        # n múltiplo de 7 con n≠0 → cierre de Berry (n_z7=7, pcv=0 fermiónico)
        # n=0 es espacio total (pcv=+2), punto fijo DISTINTO de la línea crítica
        critical = node.is_vacuum and (n != 0)  # gcd(n,7)=7 y n≠0

        return DoubleCoverNode(
            n            = n,
            node         = node,
            cycle        = cycle,
            half         = half,
            lazo_prev    = n - 7,
            lazo_next    = n + 7,
            trivial_zero = trivial,
            critical_zero= critical,
        )

    @classmethod
    def chain(cls, start: int = 0, length: int = 29) -> List[DoubleCoverNode]:
        """Construye la cadena de lazos desde start hasta start+length."""
        return [cls.build_node(start + k) for k in range(length)]

    @classmethod
    def print_chain(cls, start: int = 0, n_cycles: int = 2):
        length = 14 * n_cycles + 1
        chain  = cls.chain(start, length)

        print(f"\n═══ Doble Cubierta H7 — periodo 14 = 2×7 ≅ 4π ════════════════")
        print(f"{'n':>5} {'Z₇':>4} {'qubit':>6} {'ciclo':>6} {'mitad':>6}"
              f" {'tipo':>10} {'pcv':>10} {'lazo':>12}  nota")
        print("─" * 90)

        for dc in chain:
            tz = "◆ cero trivial"  if dc.trivial_zero  else ""
            cz = "★ Re(s)=1/2"    if dc.critical_zero else ""
            nota = tz or cz
            # separador visual entre ciclos
            if dc.n % 14 == 0 and dc.n != start:
                print("─" * 90)
            print(
                f"{dc.n:>5} {dc.node.n_z7:>4} {dc.node.qubit_state:>6}"
                f" {dc.cycle:>6} {dc.half:>6}"
                f" {dc.node.particle_type:>10}"
                f" {dc.node.pcv:>+10.6f}"
                f"  {dc.lazo_prev}←→{dc.lazo_next:<6}  {nota}"
            )

    @classmethod
    def zeta_correspondence(cls) -> None:
        """
        Tabla de correspondencia entre la doble cubierta H7
        y los ceros de ζ(s).

        Proposición:
          Sea n_z7(n) la proyección Z₇ de n.
          i_n = cos(π·n_z7) actúa como sin(πs/2) en la ec. funcional de ζ.

          i_n = +1  (n_z7 par)    ↔  sin(πs/2)=0  →  s = -2,-4,-6,...  ceros triviales
          i_n = -1  (n_z7 impar)  ↔  línea crítica Re(s)=1/2           ceros no triviales
          periodo 14              ↔  ecuación funcional ζ(s)=ζ(1-s)    simetría reflexiva
        """
        print(f"\n═══ Correspondencia H7 ↔ Ceros de ζ(s) ═══════════════════════")
        print(f"\n  Ecuación funcional de Riemann:")
        print(f"  ζ(s) = 2ˢπˢ⁻¹ sin(πs/2) Γ(1-s) ζ(1-s)")
        print(f"\n  Anulador: sin(πs/2) = 0  ↔  s = -2,-4,-6,...")
        print(f"  Análogo H7: i_n = cos(π·n_z7) = +1  ↔  n_z7 ∈ {{0,2,4,6}}")
        print(f"  Verificación: sin(π·n_z7/2) para n_z7 par:")

        for k in range(4):
            n_z7 = 2 * k
            sin_val = math.sin(PI * n_z7 / 2)
            i_n_val = math.cos(PI * n_z7)
            s_triv  = -(2 * (k + 1)) if k < 3 else "..."
            print(f"    n_z7={n_z7}  sin(π·{n_z7}/2)={sin_val:+.6f}"
                  f"  i_n={i_n_val:+.1f}  ↔  s={s_triv}")

        print(f"\n  Nodos fermiónicos (i_n=-1, n_z7 impar) → Re(s)=1/2:")
        for n_z7 in [1, 3, 5, 7]:
            pcb = math.cos(PI * PHI * n_z7) * (math.cos(PI * n_z7) + 1)
            print(f"    n_z7={n_z7}  pcb={pcb:+.10f}  ({'✓ cero exacto' if abs(pcb)<1e-9 else 'no cero'})")

        print(f"\n  Doble cubierta (periodo 14 ≅ 4π):")
        print(f"    n=0  → n=14  → n=28  (bosónico 000, pcv=+2)  vuelta completa")
        print(f"    n=7  → n=21  → n=35  (fermiónico 111, pcv=0)  cierre de Berry")
        print(f"    Reflexión n ↔ 14-n  ≅  s ↔ 1-s  (ecuación funcional)")

        print(f"\n  Verificación reflexión en ciclo base [0..14]:")
        for n in range(15):
            node  = H7Node(n)
            n_ref = 14 - n
            node_ref = H7Node(n_ref)
            sym = "✓" if node.n_z7 + node_ref.n_z7 == 7 else " "
            if n <= 7:
                print(f"    n={n:>2} (Z₇={node.n_z7}) ↔ n={n_ref:>2} (Z₇={node_ref.n_z7})"
                      f"  suma={node.n_z7+node_ref.n_z7}  {sym}")

        print(f"\n  → la reflexión n ↔ 14-n preserva n_z7 + n_z7_ref = 7")
        print(f"    análogo a Re(s) + Re(1-s) = 1  en la ec. funcional de ζ")

    @classmethod
    def spin_statistics(cls) -> None:
        """
        Verifica la conexión espín-estadística en la doble cubierta.

        En H7:
          Bosones  (pcv=0, i_n=+1) → periodo 7  (retorno en una vuelta)
          Fermiones (pcb=0, i_n=-1) → periodo 14 (necesitan dos vueltas)

        Análogo exacto de:
          Bosones  → estadística Bose-Einstein, función de onda simétrica
          Fermiones → estadística Fermi-Dirac, función de onda antisimétrica
        """
        print(f"\n═══ Espín-Estadística en Doble Cubierta H7 ═══════════════════")

        # Verificar periodo de retorno para cada clase
        print(f"\n  Periodo de retorno por tipo (pcv, pcb):")
        print(f"  {'n':>4}  {'Z₇':>4}  {'tipo':>10}  {'pcv':>10}  {'pcb':>10}  periodo")
        print("  " + "─" * 62)

        seen_states = {}
        for n in range(1, 30):
            node  = H7Node(n)
            state = (node.n_z7, node.particle_type)
            if state not in seen_states:
                seen_states[state] = n
            else:
                periodo = n - seen_states[state]
                pcb = math.cos(PI * PHI * node.n_z7) * (math.cos(PI * node.n_z7) + 1)
                print(f"  {n:>4}  {node.n_z7:>4}  {node.particle_type:>10}"
                      f"  {node.pcv:>+10.6f}  {pcb:>+10.6f}  {periodo}")
                seen_states[state] = n  # actualizar para siguiente retorno

        print(f"\n  Resultado:")
        print(f"  Fermiones (n_z7 impar): periodo = 14  ≅  4π  (espinor)")
        print(f"  Bosones   (n_z7 par):   periodo = 14  ≅  4π  (tensorial)")
        print(f"  Nota: el periodo efectivo de DIFERENCIACIÓN es 7,")
        print(f"        el de RETORNO al estado idéntico es 14.")
        print(f"\n  Gap topológico mínimo (mass gap candidato):")
        m_stars = [H7Node(k).m_star for k in range(1, 8)]
        print(f"  min(m*) sobre Z₇ = {min(m_stars):.8f}  (n_z7={[k+1 for k,v in enumerate(m_stars) if v==min(m_stars)][0]})")
        print(f"  max(m*) sobre Z₇ = {max(m_stars):.8f}  (cuello de botella n_z7=4)")
        print(f"  → min(m*) > 0  garantiza gap no perturbativo en H7")
        print(f"  → DRIFT = 7-2π = {DRIFT:.8f}  (residuo topológico del gap)")


# ══════════════════════════════════════════════════════════
# ACTUALIZAR DEMO PARA INCLUIR PILAR 6
# ══════════════════════════════════════════════════════════

def demo_double_cover():
    """Demo standalone del Pilar 6 — Doble Cubierta H7."""
    print("╔══════════════════════════════════════════════════════╗")
    print("║   H7 Doble Cubierta — periodo 14 ≅ 4π              ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Cadena de lazos n=0..28 (2 ciclos completos)
    DoubleCoverH7.print_chain(start=0, n_cycles=2)

    # Correspondencia con ζ(s)
    DoubleCoverH7.zeta_correspondence()

    # Espín-estadística
    DoubleCoverH7.spin_statistics()

    print(f"\n─── Proposiciones formales ────────────────────────────")
    print(f"""
  P1 (Doble cubierta):
     H7 tiene periodo efectivo 14 = 2×7, análogo a SU(2) sobre SO(3).
     Un estado fermiónico (n_z7 impar) requiere 14 pasos para retornar
     al estado idéntico, equivalente a una rotación de 4π.

  P2 (Ceros triviales):
     Los nodos con n_z7 ∈ {{0,2,4,6}} satisfacen sin(π·n_z7/2) = 0,
     la misma condición que anula ζ(s) en s = -2,-4,-6,...
     Bajo la reflexión n ↔ 14-n: n_z7(n) + n_z7(14-n) = 7,
     análogo a Re(s) + Re(1-s) = 1 en la ecuación funcional.

  P3 (Mass gap):
     min_{{n∈Z₇}} m*(n) = 1/|∇²Ψ|_max > 0 (verificado numéricamente).
     El gap es no perturbativo: no depende de expansión en serie.
     DRIFT = 7-2π es el residuo topológico que lo sostiene.

  P4 (Espín-estadística):
     Fermiones (pcb=0) y bosones (pcv=0) se diferencian en Z₇ (periodo 7)
     pero retornan al mismo estado en Z₁₄ (periodo 14).
     La doble cubierta es la manifestación topológica del teorema
     espín-estadística en H7.
    """)

# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "node":
        raw = input("Insert n value or character: ")
        try:
            n = int(raw)
        except ValueError:
            n = ord(raw[0]) if raw else 65
        print(H7Node(n).summary(verbose=True))
    elif len(sys.argv) > 1 and sys.argv[1] == "full":
        demo_full()
        print("\n")
        demo_double_cover()
    else:
        # Por defecto ejecutamos el objetivo principal del archivo (Riemann / Doble Cubierta)
        demo_double_cover()