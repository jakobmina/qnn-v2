"""
tests/test_h7_bridge.py
=======================
Pytest para h7_bridge.py

Cubre:
  - su2_to_cuaternion: norma unitaria del cuaternión
  - h7_to_metriplectic_state: correctitud del mapeo psi/v/energy
  - statevector_to_estado_cuantico: dimensión y valores
  - torsion_from_h7: DRIFT_072, entropía de Shannon, rangos físicos
  - qnn_grid_from_z7: pares Z₇, learning_rate áureo
  - export_binary: layout de 152 bytes (19 doubles little-endian)
  - export_json: estructura de claves y tipos
  - run_h7_bridge: integración end-to-end
"""

import math
import struct
import json
import os
import tempfile
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h7_bridge import (
    su2_to_cuaternion,
    h7_to_metriplectic_state,
    statevector_to_estado_cuantico,
    torsion_from_h7,
    qnn_grid_from_z7,
    structs_to_dict,
    export_json,
    export_binary,
    run_h7_bridge,
    MetriplecticState,
    TorsionObservables,
    EstadoCuantico,
    QNNGrid,
)

π = math.pi
φ = (1 + math.sqrt(5)) / 2
DRIFT_072 = 7 - 2 * π


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_su2(k: int = 3, angle: float = 0.5):
    """Genera una matriz SU(2) de prueba equivalente a get_su2_matrix."""
    axis = np.array([math.sin(k), math.cos(k), math.sin(k * φ)])
    axis /= np.linalg.norm(axis)
    half = angle * (π / 4) / 2
    w, (x, y, z) = math.cos(half), math.sin(half) * axis
    return np.array([[w + 1j*z, y + 1j*x], [-y + 1j*x, w - 1j*z]], dtype=complex)

DUMMY_SU2 = make_su2()
DUMMY_SV_2Q = np.array([0.5+0j, 0.5+0j, 0.5+0j, 0.5+0j])   # statevector 2-qubit
DUMMY_SV_8  = np.array([0.125]*8, dtype=float)                # statevector 3-qubit (char)
DUMMY_PROBS = {"00": 0.5, "01": 0.25, "10": 0.15, "11": 0.10}

N_SAMPLES = [1, 7, 14, 45, 65]   # varios n de prueba


# ─── 1. su2_to_cuaternion ─────────────────────────────────────────────────────

class TestSU2ToQuaternion:

    @pytest.mark.parametrize("n", N_SAMPLES)
    def test_quaternion_unit_norm(self, n):
        """El cuaternión extraído de SU(2) debe tener norma 1."""
        su2 = make_su2(k=n % 7)
        q = su2_to_cuaternion(su2)
        norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        assert math.isclose(norm, 1.0, rel_tol=1e-9), \
            f"Norma del cuaternión = {norm:.10f} para n={n}, esperada 1.0"

    def test_identity_matrix_gives_identity_quaternion(self):
        """La matriz identidad SU(2) debe dar q = (1, 0, 0, 0)."""
        identity = np.array([[1+0j, 0+0j], [0+0j, 1+0j]])
        q = su2_to_cuaternion(identity)
        assert math.isclose(abs(q.w), 1.0, rel_tol=1e-9)
        assert math.isclose(q.x, 0.0, abs_tol=1e-9)
        assert math.isclose(q.y, 0.0, abs_tol=1e-9)
        assert math.isclose(q.z, 0.0, abs_tol=1e-9)


# ─── 2. h7_to_metriplectic_state ─────────────────────────────────────────────

class TestMetriplecticState:

    @pytest.mark.parametrize("n", N_SAMPLES)
    def test_psi_is_cos_pi_phi_n(self, n):
        """ms.psi debe ser cos(πφn) (cuasi-período)."""
        ms = h7_to_metriplectic_state(n, DUMMY_SU2, DUMMY_SV_2Q)
        expected = math.cos(π * φ * n)
        assert math.isclose(ms.psi, expected, rel_tol=1e-9)

    @pytest.mark.parametrize("n", N_SAMPLES)
    def test_v_is_cos_pi_n(self, n):
        """ms.v debe ser cos(πn) (paridad)."""
        ms = h7_to_metriplectic_state(n, DUMMY_SU2, DUMMY_SV_2Q)
        expected = math.cos(π * n)
        assert math.isclose(ms.v, expected, rel_tol=1e-9)

    @pytest.mark.parametrize("n", N_SAMPLES)
    def test_energy_is_psi_classifier(self, n):
        """ms.energy debe ser psi*v + v (clasificador Ψn)."""
        ms = h7_to_metriplectic_state(n, DUMMY_SU2, DUMMY_SV_2Q)
        o_n = math.cos(π * n)
        i_n = math.cos(π * φ * n)
        expected = o_n * i_n + i_n
        assert math.isclose(ms.energy, expected, rel_tol=1e-9)

    @pytest.mark.parametrize("n", N_SAMPLES)
    def test_quaternion_is_unit(self, n):
        """El cuaternión en MetriplecticState debe tener norma 1."""
        ms = h7_to_metriplectic_state(n, DUMMY_SU2, DUMMY_SV_2Q)
        norm = math.sqrt(ms.q.w**2 + ms.q.x**2 + ms.q.y**2 + ms.q.z**2)
        assert math.isclose(norm, 1.0, rel_tol=1e-9)


# ─── 3. statevector_to_estado_cuantico ───────────────────────────────────────

class TestEstadoCuantico:

    def test_numeric_mode_has_8_psi_values(self):
        """En modo numérico, psi debe tener exactamente 8 valores."""
        ec = statevector_to_estado_cuantico(DUMMY_SV_2Q, is_char=False)
        assert len(list(ec.psi)) == 8

    def test_char_mode_maps_8_amplitudes_directly(self):
        """En modo carácter, los 8 valores reales del SV se copian directamente."""
        ec = statevector_to_estado_cuantico(DUMMY_SV_8, is_char=True)
        for i in range(8):
            assert math.isclose(ec.psi[i], float(DUMMY_SV_8[i]), rel_tol=1e-9)

    def test_numeric_interleaves_real_imag(self):
        """En modo numérico: psi[0]=Re(α0), psi[1]=Im(α0), psi[2]=Re(α1)..."""
        sv = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        ec = statevector_to_estado_cuantico(sv, is_char=False)
        assert math.isclose(ec.psi[0], 1.0, abs_tol=1e-9)  # Re(α0)
        assert math.isclose(ec.psi[1], 2.0, abs_tol=1e-9)  # Im(α0)
        assert math.isclose(ec.psi[2], 3.0, abs_tol=1e-9)  # Re(α1)
        assert math.isclose(ec.psi[3], 4.0, abs_tol=1e-9)  # Im(α1)


# ─── 4. torsion_from_h7 ───────────────────────────────────────────────────────

class TestTorsionObservables:

    @pytest.mark.parametrize("n", N_SAMPLES)
    def test_spatial_torsion_is_drift_072(self, n):
        """spatial_torsion siempre debe ser DRIFT_072 = 7 - 2π (residuo topológico)."""
        to = torsion_from_h7(n, DUMMY_PROBS)
        assert math.isclose(to.spatial_torsion, DRIFT_072, rel_tol=1e-12), \
            f"DRIFT_072 incorrecto para n={n}: {to.spatial_torsion}"

    @pytest.mark.parametrize("n", N_SAMPLES)
    def test_energy_density_is_non_negative(self, n):
        """energy_density = |Ψn| debe ser >= 0."""
        to = torsion_from_h7(n, DUMMY_PROBS)
        assert to.energy_density >= 0.0

    @pytest.mark.parametrize("n", N_SAMPLES)
    def test_entropy_gradient_in_unit_range(self, n):
        """La entropía de Shannon normalizada debe estar en [0, 1]."""
        to = torsion_from_h7(n, DUMMY_PROBS)
        assert 0.0 <= to.entropy_gradient <= 1.0, \
            f"entropy_gradient={to.entropy_gradient:.6f} fuera de [0,1] para n={n}"

    def test_max_entropy_for_uniform_distribution(self):
        """Con distribución uniforme la entropía debe ser máxima (= 1.0)."""
        n_states = 4
        uniform_probs = {f"{i:02b}": 1/n_states for i in range(n_states)}
        to = torsion_from_h7(1, uniform_probs)
        assert math.isclose(to.entropy_gradient, 1.0, rel_tol=1e-9), \
            f"Entropía uniforme = {to.entropy_gradient:.10f}, esperada 1.0"

    def test_zero_entropy_for_deterministic_distribution(self):
        """Con una sola probabilidad = 1.0 la entropía debe ser 0."""
        deterministic = {"00": 1.0}
        to = torsion_from_h7(1, deterministic)
        assert math.isclose(to.entropy_gradient, 0.0, abs_tol=1e-9), \
            f"Entropía determinista = {to.entropy_gradient:.10f}, esperada 0.0"


# ─── 5. qnn_grid_from_z7 ─────────────────────────────────────────────────────

class TestQNNGrid:

    @pytest.mark.parametrize("n_z7", range(1, 8))
    def test_grid_has_4_layers(self, n_z7):
        """La QNNGrid siempre debe tener exactamente 4 capas."""
        grid = qnn_grid_from_z7(n_z7)
        assert len(list(grid.layers)) == 4

    def test_learning_rate_is_golden_ratio_reciprocal(self):
        """La tasa de aprendizaje debe ser 1/φ ≈ 0.618 (tasa áurea)."""
        grid = qnn_grid_from_z7(3)
        expected = 1.0 / φ
        assert math.isclose(grid.learning_rate, expected, rel_tol=1e-9), \
            f"learning_rate = {grid.learning_rate:.9f}, esperado 1/φ = {expected:.9f}"

    def test_pair_indices_are_valid_z7_nodes(self):
        """Los par_indices de cada capa deben estar en el rango [0, 7]."""
        grid = qnn_grid_from_z7(5)
        for i in range(4):
            a = grid.layers[i].pair_indices[0]
            b = grid.layers[i].pair_indices[1]
            assert 0 <= a <= 7, f"pair_indices[0]={a} fuera de rango Z7"
            assert 0 <= b <= 7, f"pair_indices[1]={b} fuera de rango Z7"

    @pytest.mark.parametrize("n_z7", range(1, 8))
    def test_weights_are_finite(self, n_z7):
        """Todos los pesos de las capas deben ser valores finitos."""
        grid = qnn_grid_from_z7(n_z7)
        for i in range(4):
            assert math.isfinite(grid.layers[i].weight), \
                f"Peso no finito en capa {i} para n_z7={n_z7}"
            assert math.isfinite(grid.layers[i].bias), \
                f"Bias no finito en capa {i} para n_z7={n_z7}"


# ─── 6. export_binary ─────────────────────────────────────────────────────────

class TestExportBinary:

    def test_binary_file_is_152_bytes(self, tmp_path):
        """El binario exportado debe tener exactamente 152 bytes (19 doubles × 8 bytes)."""
        ms = h7_to_metriplectic_state(45, DUMMY_SU2, DUMMY_SV_2Q)
        to = torsion_from_h7(45, DUMMY_PROBS)
        ec = statevector_to_estado_cuantico(DUMMY_SV_2Q, is_char=False)

        path = str(tmp_path / "test_export.bin")
        export_binary(ms, to, ec, path=path)

        assert os.path.exists(path)
        size = os.path.getsize(path)
        assert size == 152, f"Tamaño del binario = {size} bytes, esperado 152"

    def test_binary_roundtrip_psi_value(self, tmp_path):
        """El valor psi de MetriplecticState debe sobrevivir el ciclo export/read."""
        ms = h7_to_metriplectic_state(45, DUMMY_SU2, DUMMY_SV_2Q)
        to = torsion_from_h7(45, DUMMY_PROBS)
        ec = statevector_to_estado_cuantico(DUMMY_SV_2Q, is_char=False)

        path = str(tmp_path / "test_roundtrip.bin")
        export_binary(ms, to, ec, path=path)

        with open(path, "rb") as f:
            buf = f.read(152)

        # psi es el primer double (offset 0)
        psi_read = struct.unpack_from("<d", buf, 0)[0]
        assert math.isclose(psi_read, ms.psi, rel_tol=1e-12)

    def test_binary_drift_072_preserved(self, tmp_path):
        """DRIFT_072 = spatial_torsion (posición 9) debe leerse exactamente."""
        ms = h7_to_metriplectic_state(1, DUMMY_SU2, DUMMY_SV_2Q)
        to = torsion_from_h7(1, DUMMY_PROBS)
        ec = statevector_to_estado_cuantico(DUMMY_SV_2Q, is_char=False)

        path = str(tmp_path / "test_drift.bin")
        export_binary(ms, to, ec, path=path)

        with open(path, "rb") as f:
            buf = f.read(152)

        # offset 9 × 8 = 72 bytes → spatial_torsion
        torsion_read = struct.unpack_from("<d", buf, 9 * 8)[0]
        assert math.isclose(torsion_read, DRIFT_072, rel_tol=1e-12)


# ─── 7. export_json ──────────────────────────────────────────────────────────

class TestExportJson:

    def test_json_has_all_top_level_keys(self, tmp_path):
        """El JSON exportado debe contener las 4 secciones principales."""
        ms = h7_to_metriplectic_state(45, DUMMY_SU2, DUMMY_SV_2Q)
        to = torsion_from_h7(45, DUMMY_PROBS)
        ec = statevector_to_estado_cuantico(DUMMY_SV_2Q, is_char=False)
        grid = qnn_grid_from_z7(3)

        data = structs_to_dict(ms, to, ec, grid)
        path = str(tmp_path / "test.json")
        export_json(data, path=path)

        with open(path) as f:
            loaded = json.load(f)

        for key in ["MetriplecticState", "TorsionObservables", "EstadoCuantico", "QNNGrid"]:
            assert key in loaded, f"Clave '{key}' ausente en el JSON"

    def test_json_estado_cuantico_has_8_psi(self, tmp_path):
        """EstadoCuantico.psi en el JSON debe tener 8 elementos."""
        ms = h7_to_metriplectic_state(45, DUMMY_SU2, DUMMY_SV_2Q)
        to = torsion_from_h7(45, DUMMY_PROBS)
        ec = statevector_to_estado_cuantico(DUMMY_SV_2Q, is_char=False)
        grid = qnn_grid_from_z7(3)

        data = structs_to_dict(ms, to, ec, grid)
        path = str(tmp_path / "test_psi.json")
        export_json(data, path=path)

        with open(path) as f:
            loaded = json.load(f)

        assert len(loaded["EstadoCuantico"]["psi"]) == 8

    def test_json_extra_metrics_included(self, tmp_path):
        """Las métricas extra (covarianza/asimetría) deben aparecer en ExtraMetrics."""
        ms = h7_to_metriplectic_state(45, DUMMY_SU2, DUMMY_SV_2Q)
        to = torsion_from_h7(45, DUMMY_PROBS)
        ec = statevector_to_estado_cuantico(DUMMY_SV_2Q, is_char=False)
        grid = qnn_grid_from_z7(3)

        extras = {"covariance_q1q2": 0.123, "asymmetry_q1q2": -0.045}
        data = structs_to_dict(ms, to, ec, grid, extra_metrics=extras)
        path = str(tmp_path / "test_extras.json")
        export_json(data, path=path)

        with open(path) as f:
            loaded = json.load(f)

        assert "ExtraMetrics" in loaded
        assert math.isclose(loaded["ExtraMetrics"]["covariance_q1q2"], 0.123, rel_tol=1e-9)


# ─── 8. run_h7_bridge (integración) ──────────────────────────────────────────

class TestRunH7Bridge:

    @pytest.mark.parametrize("n", [1, 7, 45])
    def test_bridge_returns_all_struct_keys(self, n, tmp_path):
        """run_h7_bridge debe devolver dict con las 5 claves esperadas."""
        result = run_h7_bridge(
            n=n, su2_matrix=DUMMY_SU2, statevector=DUMMY_SV_2Q,
            probabilities=DUMMY_PROBS,
            export_path=str(tmp_path / f"test_n{n}"),
            export_format="none"
        )
        for key in ["MetriplecticState", "TorsionObservables",
                    "EstadoCuantico", "QNNGrid", "dict"]:
            assert key in result, f"Clave '{key}' ausente en resultado del bridge"

    def test_bridge_creates_json_and_bin(self, tmp_path):
        """Con export_format='both', deben crearse los dos archivos."""
        base = str(tmp_path / "h7_test")
        run_h7_bridge(
            n=45, su2_matrix=DUMMY_SU2, statevector=DUMMY_SV_2Q,
            probabilities=DUMMY_PROBS,
            export_path=base,
            export_format="both"
        )
        assert os.path.exists(base + ".json"), "Falta el archivo .json"
        assert os.path.exists(base + ".bin"),  "Falta el archivo .bin"

    def test_bridge_char_mode_uses_8d_statevector(self, tmp_path):
        """En modo carácter (is_char=True), EstadoCuantico.psi se llena con el SV de 8 dims."""
        result = run_h7_bridge(
            n=65, su2_matrix=DUMMY_SU2, statevector=DUMMY_SV_8,
            probabilities=DUMMY_PROBS,
            export_path=str(tmp_path / "test_char"),
            export_format="none",
            is_char=True
        )
        ec = result["EstadoCuantico"]
        for i in range(8):
            assert math.isclose(ec.psi[i], float(DUMMY_SV_8[i]), rel_tol=1e-9)
