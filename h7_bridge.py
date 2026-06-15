"""
h7_bridge.py — Bridge bidireccional H7 Python ↔ C metriplectic
Exporta los estados del pipeline H7 al formato exacto de metriplectic.h
"""
import math, struct, json, ctypes
import numpy as np

pi = math.pi
phi = (1 + math.sqrt(5)) / 2  # Razón Áurea ≈ 1.618033...


# ────────────────────────────────────────────
# 1. CTYPES MIRROR de metriplectic.h
# ────────────────────────────────────────────

class Cuaternion(ctypes.Structure):
    _fields_ = [("w", ctypes.c_double), ("x", ctypes.c_double),
                ("y", ctypes.c_double), ("z", ctypes.c_double)]

class MetriplecticState(ctypes.Structure):
    _fields_ = [("psi",    ctypes.c_double),
                ("v",      ctypes.c_double),
                ("energy", ctypes.c_double),
                ("q",      Cuaternion)]

class TorsionObservables(ctypes.Structure):
    _fields_ = [("energy_density",   ctypes.c_double),
                ("entropy_gradient", ctypes.c_double),
                ("spatial_torsion",  ctypes.c_double),
                ("chirality",        ctypes.c_double)]

class EstadoCuantico(ctypes.Structure):
    _fields_ = [("psi", ctypes.c_double * 8)]

class QNNLayer(ctypes.Structure):
    _fields_ = [("weight",       ctypes.c_double),
                ("bias",         ctypes.c_double),
                ("pair_indices", ctypes.c_int * 2)]

class QNNGrid(ctypes.Structure):
    _fields_ = [("layers",        QNNLayer * 4),
                ("learning_rate", ctypes.c_double)]


# ────────────────────────────────────────────
# 2. FUNCIONES DE CONVERSIÓN Python → C struct
# ────────────────────────────────────────────

def su2_to_cuaternion(su2_matrix: np.ndarray) -> Cuaternion:
    """
    Extrae cuaternión (w,x,y,z) de una matriz SU(2) 2×2:
        M = [[w+iz,  y+ix],
             [-y+ix, w-iz]]
    """
    w = float(np.real(su2_matrix[0, 0]))
    z = float(np.imag(su2_matrix[0, 0]))
    x = float(np.imag(su2_matrix[0, 1]))
    y = float(np.real(su2_matrix[0, 1]))
    # renormalizar por seguridad numérica
    norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    return Cuaternion(w=w, x=x, y=y, z=z)


def h7_to_metriplectic_state(n: int,
                              su2_matrix: np.ndarray,
                              statevector: np.ndarray) -> MetriplecticState:
    """
    Construye MetriplecticState desde los datos del pipeline H7.

    Mapping semántico:
      psi    ← cos(πφn)  [cuasi-período = amplitud de fase]
      v      ← cos(πn)   [paridad = velocidad de flujo]
      energy ← Ψn = psi*v + v  [clasificador de partícula]
    """
    o_n = math.cos(pi * n)
    i_n = math.cos(pi * phi * n)
    energy = o_n * i_n + i_n
    q = su2_to_cuaternion(su2_matrix)
    return MetriplecticState(psi=i_n, v=o_n, energy=energy, q=q)


def statevector_to_estado_cuantico(sv: np.ndarray, is_char: bool = False) -> EstadoCuantico:
    """
    Empaqueta un statevector complejo de Qiskit en psi[8] real.
    Si is_char es True, toma directamente las 8 partes reales (para UTF-8 QNN).
    Convención original: [Re(α0), Im(α0), Re(α1), Im(α1), Re(α2), Im(α2), Re(α3), Im(α3)]
    """
    ec = EstadoCuantico()
    if is_char and len(sv) == 8:
        for i in range(8):
            ec.psi[i] = float(np.real(sv[i]))
    else:
        flat = []
        for amp in sv[:4]:           # máx 4 amplitudes del circuito de 2q
            flat.append(float(np.float64((amp.real))))   # Re(αk) primero
            flat.append(float(np.float64((amp.imag))))   # Im(αk) después
        flat = (flat + [0.0] * 8)[:8]
        for i, v in enumerate(flat):
            ec.psi[i] = v
    return ec


def torsion_from_h7(n: int, probabilities: dict) -> TorsionObservables:
    """
    Deriva TorsionObservables desde los observables H7.

    Mapping físico:
      energy_density   ← |Ψn|   (módulo del clasificador)
      entropy_gradient ← -Σ p·log(p)  (entropía de Shannon del circuito)
      spatial_torsion  ← DRIFT_072 = 7 - 2π  (residuo topológico)
      chirality        ← cos(πn)  (paridad → helicidad)
    """
    DRIFT_072 = 7 - 2 * pi

    o_n = math.cos(pi * n)
    i_n = math.cos(pi * phi * n)
    psi_n = abs(o_n * i_n + i_n)

    # entropía de Shannon de las probabilidades del circuito
    entropy = 0.0
    for p in probabilities.values():
        if p > 0:
            entropy -= p * math.log2(p)
    # normalizar al rango [0,1]
    max_entropy = math.log2(max(len(probabilities), 1))
    entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0

    return TorsionObservables(
        energy_density=psi_n,
        entropy_gradient=entropy_norm,
        spatial_torsion=DRIFT_072,
        chirality=o_n
    )


def qnn_grid_from_z7(n_z7: int) -> QNNGrid:
    """
    Inicializa QNNGrid con los pares complementarios Z₇.
    Pares antagonistas: (1,6)→i, (2,5)→j, (3,4)→k  (Q₈)
    Los pair_indices mapean al índice Z₇ del nodo.
    """
    Z7_PAIRS = [(1, 6), (2, 5), (3, 4), (0, 7 % 7)]
    PHI_LEARNING = 1.0 / phi  # tasa de aprendizaje áurea

    grid = QNNGrid()
    grid.learning_rate = PHI_LEARNING

    for i, (a, b) in enumerate(Z7_PAIRS):
        weight = math.cos(pi * phi * a) * math.cos(pi * phi * b)
        bias   = math.cos(pi * a) + math.cos(pi * b)
        grid.layers[i].weight = weight
        grid.layers[i].bias   = bias
        grid.layers[i].pair_indices[0] = a
        grid.layers[i].pair_indices[1] = b

    return grid


# ────────────────────────────────────────────
# 3. SERIALIZACIÓN (JSON + Binary)
# ────────────────────────────────────────────

def structs_to_dict(ms: MetriplecticState, to: TorsionObservables,
                    ec: EstadoCuantico, grid: QNNGrid, extra_metrics: dict = None) -> dict:
    """Serializa todas las structs a dict Python (para JSON export)."""
    data = {
        "MetriplecticState": {
            "psi":    ms.psi,
            "v":      ms.v,
            "energy": ms.energy,
            "q": {"w": ms.q.w, "x": ms.q.x, "y": ms.q.y, "z": ms.q.z}
        },
        "TorsionObservables": {
            "energy_density":   to.energy_density,
            "entropy_gradient": to.entropy_gradient,
            "spatial_torsion":  to.spatial_torsion,
            "chirality":        to.chirality
        },
        "EstadoCuantico": {
            "psi": list(ec.psi)
        },
        "QNNGrid": {
            "learning_rate": grid.learning_rate,
            "layers": [
                {
                    "weight": grid.layers[i].weight,
                    "bias":   grid.layers[i].bias,
                    "pair_indices": [grid.layers[i].pair_indices[0],
                                     grid.layers[i].pair_indices[1]]
                }
                for i in range(4)
            ]
        }
    }
    if extra_metrics:
        data["ExtraMetrics"] = extra_metrics
    return data


def export_json(data: dict, path: str = "h7_state.json"):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[h7_bridge] JSON exportado → {path}")


def export_binary(ms: MetriplecticState, to: TorsionObservables,
                  ec: EstadoCuantico, path: str = "h7_state.bin"):
    """
    Exporta las structs en binario little-endian.
    Layout: MetriplecticState (7 doubles) | TorsionObservables (4 doubles) | EstadoCuantico (8 doubles)
    Total: 19 × 8 bytes = 152 bytes
    """
    fmt = "<" + "d" * 7 + "d" * 4 + "d" * 8
    data = struct.pack(fmt,
        ms.psi, ms.v, ms.energy, ms.q.w, ms.q.x, ms.q.y, ms.q.z,
        to.energy_density, to.entropy_gradient, to.spatial_torsion, to.chirality,
        *list(ec.psi)
    )
    with open(path, "wb") as f:
        f.write(data)
    print(f"[h7_bridge] Binary exportado → {path}  ({len(data)} bytes)")


# ────────────────────────────────────────────
# 4. FUNCIÓN PRINCIPAL DE INTEGRACIÓN
# ────────────────────────────────────────────

def run_h7_bridge(n: int,
                  su2_matrix: np.ndarray,
                  statevector: np.ndarray,
                  probabilities: dict,
                  export_path: str = "h7_state",
                  export_format: str = "both",
                  extra_metrics: dict = None,
                  is_char: bool = False) -> dict:
    """
    Punto de entrada unificado.
    Recibe los outputs del script H7 y devuelve todas las structs C populadas.

    Args:
        n            : índice original (antes de proyección Z₇)
        su2_matrix   : matriz SU(2) 2×2 del modo 'dynamic'
        statevector  : statevector de Qiskit AER (array complejo)
        probabilities: dict {estado_binario: prob} del sampler
        export_path  : base del path de salida (sin extensión)
        export_format: "json" | "binary" | "both" | "none"
        extra_metrics: dict opcional de métricas extra (ej. covarianza)
        is_char      : Flag indicando si statevector proviene de un carácter UTF-8

    Returns:
        dict con todas las structs y el dict serializable
    """
    n_z7 = 7 if (n % 7 == 0 and n != 0) else (n % 7)

    ms   = h7_to_metriplectic_state(n, su2_matrix, statevector)
    to   = torsion_from_h7(n, probabilities)
    ec   = statevector_to_estado_cuantico(statevector, is_char)
    grid = qnn_grid_from_z7(n_z7)

    data = structs_to_dict(ms, to, ec, grid, extra_metrics)

    if export_format in ("json", "both"):
        export_json(data, path=export_path + ".json")
    if export_format in ("binary", "both"):
        export_binary(ms, to, ec, path=export_path + ".bin")

    return {"MetriplecticState": ms, "TorsionObservables": to,
            "EstadoCuantico": ec, "QNNGrid": grid, "dict": data}


if __name__ == "__main__":
    def run_standalone_test():
        print("[h7_bridge] Test de exportación autónomo")
        
        # 1) Datos de ejemplo (pueden venir de cualquier n)
        n_test = 12
        su2_example = np.array([
            [0.9793+0.0072j, -0.0316-0.0691j],
            [0.0067-0.0074j, 0.9774+0.0684j]
        ])
        # Simular un statevector (Qiskit 2x2 en este caso)
        sv_example = np.array([0.9971+0.0072j, -0.0316-0.0691j], dtype=complex)
        probs_example = {'00': 0.6, '01': 0.2, '10': 0.15, '11': 0.05}
        
        # 2) Generar métricas extra (covarianza, etc.)
        extra = {
            "covariance": -0.007565,
            "asymmetry": 0.028320
        }
        
        # 3) Ejecutar el bridge
        out = run_h7_bridge(
            n=n_test,
            su2_matrix=su2_example,
            statevector=sv_example,
            probabilities=probs_example,
            export_path="his-torial/_demo_standalone",
            export_format="both",
            extra_metrics=extra
        )
        
        ms = out["MetriplecticState"]
        to = out["TorsionObservables"]
        
        print(f"[h7_bridge] Check final:")
        print(f"  MetriplecticState: psi={ms.psi}, energy={ms.energy}")
        print(f"  TorsionObservables: density={to.energy_density}, torsion={to.spatial_torsion}")
        print("  Se generaron demo/h7_demo_standalone.json y .bin")

    run_standalone_test()