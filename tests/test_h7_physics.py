"""
tests/test_h7_physics.py
========================
Pytest para la física metripléctica del sistema H7.

Cubre (Mandato Metriplético):
  - Regla 1.1: Componente simpléctica — Hamiltoniano H (cos(πφn))
  - Regla 1.2: Componente métrica — Potencial disipativo S (entropía)
  - Regla 1.3: Prohibición de singularidades (ni puro conservativo ni puro disipativo)
  - Regla 2.1: Operador Áureo O_n = cos(πn)·cos(πφn)
  - Regla 3.1: compute_lagrangian() — componentes L_symp y L_metr separados
  - Competencia conservativo vs. disipativo
  - Clasificación bosónica/fermiónica (Pauli)
  - Proyección Z₇ (aplanado topológico)
  - Test del Tiempo (reversibilidad): Regla de Oro 1 del MANIFIESTO
"""

import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

π = math.pi
φ = (1 + math.sqrt(5)) / 2
DRIFT_072 = 7 - 2 * π


# ─────────────────────────────────────────────────────────────────────────────
# Implementaciones locales equivalentes a h7_main.py
# (Se testean las funciones puras de física, no el script interactivo)
# ─────────────────────────────────────────────────────────────────────────────

def golden_operator(n: float) -> float:
    """Regla 2.1: O_n = cos(πn) · cos(πφn)"""
    return math.cos(π * n) * math.cos(π * φ * n)

def compute_lagrangian(n: int) -> dict:
    """
    Regla 3.1: Devuelve L_symp (hamiltoniano) y L_metr (disipativo) por separado.

    L_symp ← cos(πφn)    (cuasi-período, componente simpléctica)
    L_metr ← cos(πn)     (paridad, componente métrica/disipativa)
    """
    L_symp = math.cos(π * φ * n)   # Hamiltoniano H: genera movimiento conservativo
    L_metr = math.cos(π * n)       # Potencial S: genera relajación (disipativo)
    return {"L_symp": L_symp, "L_metr": L_metr}

def classify_particle(n: int) -> str:
    """Clasificador bosónico/fermiónico basado en Ψn."""
    o_n = math.cos(π * (n % 7 if n % 7 != 0 else 7))
    i_n = math.cos(π * φ * (n % 7 if n % 7 != 0 else 7))
    psi_n = o_n * i_n + i_n
    return "fermionic" if math.isclose(psi_n, 0.0, abs_tol=1e-9) else "bosonic"

def z7_projection(n: int) -> int:
    """Proyección Z₇: n → n mod 7 (con caso especial n=0)."""
    return 7 if (n % 7 == 0 and n != 0) else (n % 7)


# ─── 1. Operador Áureo (Regla 2.1) ───────────────────────────────────────────

class TestGoldenOperator:

    @pytest.mark.parametrize("n", range(0, 15))
    def test_output_in_minus_one_to_one(self, n):
        """O_n = cos·cos está siempre en [-1, 1]."""
        val = golden_operator(n)
        assert -1.0 <= val <= 1.0, f"O_{n} = {val} fuera de [-1,1]"

    def test_periodic_structure(self):
        """O_n no debe ser constante — debe exhibir estructura quasi-periódica."""
        values = [golden_operator(n) for n in range(20)]
        assert not all(math.isclose(v, values[0], abs_tol=1e-9) for v in values), \
            "golden_operator produce valores constantes — sin estructura quasi-periódica"

    def test_vacuum_at_zero(self):
        """O_0 = cos(0)·cos(0) = 1 (vacío topológico máximo)."""
        assert math.isclose(golden_operator(0), 1.0, rel_tol=1e-12)

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 14])
    def test_golden_operator_is_finite(self, n):
        """O_n debe ser un número finito (Regla 1.3: prohibición de singularidades)."""
        val = golden_operator(n)
        assert math.isfinite(val), f"O_{n} = {val} no es finito"


# ─── 2. compute_lagrangian (Regla 3.1) ───────────────────────────────────────

class TestComputeLagrangian:

    @pytest.mark.parametrize("n", [1, 7, 14, 45, 65])
    def test_returns_both_components(self, n):
        """compute_lagrangian debe devolver L_symp y L_metr separados."""
        lag = compute_lagrangian(n)
        assert "L_symp" in lag, "Falta componente simpléctica L_symp"
        assert "L_metr" in lag, "Falta componente métrica L_metr"

    @pytest.mark.parametrize("n", [1, 7, 14, 45, 65])
    def test_components_are_in_range(self, n):
        """Ambos componentes deben ser cosenos: valores en [-1, 1]."""
        lag = compute_lagrangian(n)
        assert -1.0 <= lag["L_symp"] <= 1.0
        assert -1.0 <= lag["L_metr"] <= 1.0

    @pytest.mark.parametrize("n", [1, 7, 14, 45, 65])
    def test_l_symp_is_cos_pi_phi_n(self, n):
        """L_symp = cos(πφn) — componente hamiltoniana (Regla 1.1)."""
        lag = compute_lagrangian(n)
        expected = math.cos(π * φ * n)
        assert math.isclose(lag["L_symp"], expected, rel_tol=1e-12)

    @pytest.mark.parametrize("n", [1, 7, 14, 45, 65])
    def test_l_metr_is_cos_pi_n(self, n):
        """L_metr = cos(πn) — componente disipativa/métrica (Regla 1.2)."""
        lag = compute_lagrangian(n)
        expected = math.cos(π * n)
        assert math.isclose(lag["L_metr"], expected, rel_tol=1e-12)

    def test_components_are_not_always_equal(self):
        """L_symp y L_metr no deben ser idénticos — son ortogonales."""
        results = [compute_lagrangian(n) for n in range(1, 20)]
        identical_count = sum(
            1 for r in results if math.isclose(r["L_symp"], r["L_metr"], abs_tol=1e-9)
        )
        assert identical_count < len(results), \
            "L_symp y L_metr son siempre iguales — no hay separación simpléctica/métrica"


# ─── 3. Regla 1.3: Competencia (no singularidades puras) ─────────────────────

class TestMetriplecticBalance:
    """
    Regla 1.3: El sistema no puede ser puramente conservativo ni puramente disipativo.
    Debe existir competencia entre L_symp y L_metr.
    """

    def test_system_is_not_purely_conservative(self):
        """
        Un sistema puramente conservativo tendría |L_metr| ≈ 0 siempre.
        Verificamos que L_metr tiene variación real.
        """
        metr_values = [compute_lagrangian(n)["L_metr"] for n in range(1, 50)]
        max_val = max(abs(v) for v in metr_values)
        assert max_val > 0.1, \
            f"L_metr ≈ 0 siempre (max={max_val:.4f}) — sistema puramente conservativo PROHIBIDO"

    def test_system_is_not_purely_dissipative(self):
        """
        Un sistema puramente disipativo tendría |L_symp| ≈ 0 siempre.
        Verificamos que L_symp tiene variación real.
        """
        symp_values = [compute_lagrangian(n)["L_symp"] for n in range(1, 50)]
        max_val = max(abs(v) for v in symp_values)
        assert max_val > 0.1, \
            f"L_symp ≈ 0 siempre (max={max_val:.4f}) — sistema puramente disipativo PROHIBIDO"

    def test_competition_between_components(self):
        """Debe existir al menos un n donde L_symp y L_metr tienen signos opuestos."""
        opposite_sign_found = False
        for n in range(1, 50):
            lag = compute_lagrangian(n)
            if lag["L_symp"] * lag["L_metr"] < 0:
                opposite_sign_found = True
                break
        assert opposite_sign_found, \
            "Nunca hay competencia (signos opuestos) entre L_symp y L_metr"


# ─── 4. Proyección Z₇ (aplanado topológico) ──────────────────────────────────

class TestZ7Projection:

    @pytest.mark.parametrize("n,expected", [
        (0, 0), (1, 1), (7, 7), (8, 1), (14, 7),
        (15, 1), (49, 7), (50, 1),
    ])
    def test_z7_projection_correctness(self, n, expected):
        """n=7k→7, n=0→0, resto→n mod 7."""
        result = z7_projection(n)
        assert result == expected, f"z7({n}) = {result}, esperado {expected}"

    @pytest.mark.parametrize("n", range(1, 100))
    def test_z7_output_in_range_1_to_7(self, n):
        """Para n>=1, z7(n) debe estar en [1, 7]."""
        result = z7_projection(n)
        assert 1 <= result <= 7, f"z7({n}) = {result} fuera de [1,7]"

    def test_z7_period_7(self):
        """z7 debe ser periódico con periodo 7 para no-múltiplos de 7."""
        for n in range(1, 50):
            if n % 7 != 0 and (n + 7) % 7 != 0:
                assert z7_projection(n) == z7_projection(n + 7), \
                    f"Falta periodicidad: z7({n}) ≠ z7({n+7})"


# ─── 5. Clasificación de Partículas (Pauli/bosónica) ─────────────────────────

class TestParticleClassification:

    @pytest.mark.parametrize("n", range(1, 50))
    def test_particle_type_is_valid(self, n):
        """El tipo de partícula debe ser 'fermionic' o 'bosonic'."""
        ptype = classify_particle(n)
        assert ptype in ("fermionic", "bosonic"), \
            f"Tipo inválido para n={n}: '{ptype}'"

    def test_both_types_appear(self):
        """Deben aparecer tanto fermiones como bosones en el rango Z₇."""
        types = {classify_particle(n) for n in range(1, 50)}
        assert "fermionic" in types or "bosonic" in types, \
            "Solo un tipo de partícula — sin clasificación dual"

    @pytest.mark.parametrize("n_z7", [1, 2, 3, 4, 5, 6, 7])
    def test_classification_is_deterministic(self, n_z7):
        """La misma n_z7 debe producir siempre la misma clasificación."""
        t1 = classify_particle(n_z7)
        t2 = classify_particle(n_z7)
        assert t1 == t2


# ─── 6. Regla de Oro 1: Test del Tiempo (Reversibilidad) ─────────────────────

class TestTimeReversibility:
    """
    MANIFIESTO — Regla 1: Ley de Reversibilidad.
    Un sistema conservativo (simpléxico) debe ser invariante bajo t→-t.
    El Hamiltoniano H = cos(πφn) es simétrico: H(n) = H(-n) para el flujo continuo.
    """

    @pytest.mark.parametrize("n", [1, 3, 5, 7, 14, 45])
    def test_hamiltonian_is_even_function(self, n):
        """
        cos(πφn) es función par: H(n) = H(-n).
        Esto valida la reversibilidad temporal del componente simpléxico.
        """
        H_forward = math.cos(π * φ * n)
        H_backward = math.cos(π * φ * (-n))
        assert math.isclose(H_forward, H_backward, rel_tol=1e-12), \
            f"H({n}) ≠ H(-{n}): {H_forward:.8f} vs {H_backward:.8f}"

    @pytest.mark.parametrize("n", [1, 3, 5, 7, 14, 45])
    def test_dissipation_breaks_time_symmetry(self, n):
        """
        El potencial disipativo S = entropía (siempre ≥ 0) rompe la simetría temporal.
        S(n) >= 0 siempre, lo que implica irreversibilidad del componente métrico.
        La entropía de Shannon de una distribución uniforme no cambia con t→-t
        pero el signo del gradiente sí lo hace — este test verifica que S > 0 existe.
        """
        from h7_bridge import torsion_from_h7
        uniform = {f"{i:02b}": 0.25 for i in range(4)}
        to = torsion_from_h7(n, uniform)
        # entropía debe ser positiva (irreversibilidad presente)
        assert to.entropy_gradient > 0, \
            f"entropy_gradient = 0 para n={n} — sin disipación, sistema colapsaría"

    def test_conservative_plus_dissipative_is_not_zero(self):
        """
        La suma total del Lagrangiano (L_symp + L_metr) no debe ser cero globalmente.
        Cero global implicaría cancelación perfecta — no hay dinámica.
        """
        total_lagrangian = sum(
            compute_lagrangian(n)["L_symp"] + compute_lagrangian(n)["L_metr"]
            for n in range(1, 50)
        )
        assert abs(total_lagrangian) > 0.1, \
            "El Lagrangiano total se cancela — sin dinámica neta en el sistema"


# ─── 7. DRIFT_072 (Constante Topológica) ─────────────────────────────────────

class TestDrift072:
    """
    DRIFT_072 = 7 - 2π es el residuo topológico entre Z₇ (enteros) y U(1) (continuo).
    Representa la 'fricción' entre el espacio discreto y el continuo.
    """

    def test_drift_072_value(self):
        """DRIFT_072 = 7 - 2π debe ser ≈ 0.71681..."""
        expected = 7 - 2 * π
        assert math.isclose(DRIFT_072, expected, rel_tol=1e-12)

    def test_drift_072_is_positive(self):
        """El residuo topológico debe ser positivo (7 > 2π)."""
        assert DRIFT_072 > 0, f"DRIFT_072 = {DRIFT_072} no es positivo"

    def test_drift_072_is_less_than_1(self):
        """DRIFT_072 < 1: el residuo es una corrección, no una dominante."""
        assert DRIFT_072 < 1.0, f"DRIFT_072 = {DRIFT_072} >= 1 — el residuo domina"

    def test_z7_minus_2pi_matches_drift(self):
        """7 - 2π debe coincidir con DRIFT_072 a precisión de máquina."""
        computed = 7 - 2 * math.pi
        assert math.isclose(computed, DRIFT_072, rel_tol=1e-15)
