"""
tests/test_utf8_qnn.py
======================
Pytest para utf8_qnn_poc.py

Cubre:
  - Codificación UTF-8 → amplitudes normalizadas
  - Flag de seguridad (bit MSB forzado a 1)
  - Circuito CSWAP: encriptación y decriptación reversibles
  - Isomorfismo Nivel 3: conservación de la información (char_in == char_out)
  - Invarianza de norma del statevector (unitariedad del circuito)
"""

import math
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utf8_qnn_poc import (
    char_to_amplitudes,
    amplitudes_to_char,
    build_qnn_cipher_circuit,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

ASCII_CHARS = ["A", "Z", "a", "z", "0", "9", " ", "!"]
PRINTABLE_BYTES = [65, 90, 97, 122, 48, 57, 32, 33]  # equiv. de ASCII_CHARS


# ─── 1. char_to_amplitudes ────────────────────────────────────────────────────

class TestCharToAmplitudes:

    @pytest.mark.parametrize("char", ASCII_CHARS)
    def test_output_length_is_8(self, char):
        """Las amplitudes siempre deben ser un vector de 8 elementos."""
        amps, bits = char_to_amplitudes(char)
        assert len(amps) == 8
        assert len(bits) == 8

    @pytest.mark.parametrize("char", ASCII_CHARS)
    def test_amplitudes_are_normalized(self, char):
        """El statevector debe tener norma 1 (válido para Qiskit)."""
        amps, _ = char_to_amplitudes(char)
        norm = np.linalg.norm(amps)
        assert math.isclose(norm, 1.0, rel_tol=1e-9), \
            f"Norma = {norm:.10f} para '{char}', esperada 1.0"

    @pytest.mark.parametrize("char", ASCII_CHARS)
    def test_security_flag_msb_is_set(self, char):
        """El bit MSB (índice 0) siempre debe ser 1 (flag de integridad/paridad topológica)."""
        _, bits = char_to_amplitudes(char)
        assert bits[0] == 1, \
            f"Flag de seguridad NO activo para '{char}': bits = {bits}"

    @pytest.mark.parametrize("char", ASCII_CHARS)
    def test_amplitudes_are_non_negative(self, char):
        """Las amplitudes (derivadas de bits 0/1) deben ser >= 0."""
        amps, _ = char_to_amplitudes(char)
        assert all(a >= 0 for a in amps), \
            f"Amplitud negativa detectada para '{char}': {amps}"

    def test_distinct_chars_have_distinct_amplitudes(self):
        """Caracteres distintos deben producir amplitudes distintas."""
        amps_A, _ = char_to_amplitudes("A")
        amps_B, _ = char_to_amplitudes("B")
        assert not np.allclose(amps_A, amps_B), \
            "A y B produjeron amplitudes idénticas — colisión de codificación"


# ─── 2. amplitudes_to_char ────────────────────────────────────────────────────

class TestAmplitudesToChar:

    @pytest.mark.parametrize("char", ASCII_CHARS)
    def test_encode_decode_roundtrip(self, char):
        """Ciclo completo encode→decode debe recuperar el carácter original."""
        amps, bits = char_to_amplitudes(char)
        norm_factor = np.linalg.norm(bits)
        recovered = amplitudes_to_char(amps, norm_factor)
        assert recovered == char, \
            f"Fallo de roundtrip: '{char}' → '{recovered}'"


# ─── 3. build_qnn_cipher_circuit ─────────────────────────────────────────────

class TestCipherCircuit:

    def test_circuit_has_3_qubits(self):
        """El circuito de cifrado opera sobre exactamente 3 qubits (8 dimensiones)."""
        qc = build_qnn_cipher_circuit()
        assert qc.num_qubits == 3

    def test_circuit_operations_present(self):
        """El circuito debe contener al menos una operación H, CSWAP y CX."""
        qc = build_qnn_cipher_circuit()
        op_names = [instr.operation.name for instr in qc.data]
        assert "h" in op_names,     "Falta la compuerta Hadamard (H)"
        assert "cswap" in op_names, "Falta la compuerta CSWAP (Fredkin)"
        assert "cx" in op_names,    "Falta la compuerta CX (CNOT)"

    @pytest.mark.parametrize("char", ASCII_CHARS)
    def test_encryption_is_unitary_norm_preserving(self, char):
        """El circuito es unitario: cifrar no debe cambiar la norma del statevector."""
        amps, _ = char_to_amplitudes(char)
        sv = Statevector(amps)
        qc = build_qnn_cipher_circuit()
        sv_enc = sv.evolve(qc)
        norm_enc = np.linalg.norm(sv_enc.data)
        assert math.isclose(norm_enc, 1.0, rel_tol=1e-9), \
            f"Norma post-cifrado = {norm_enc:.10f} para '{char}', esperada 1.0"

    @pytest.mark.parametrize("char", ASCII_CHARS)
    def test_decrypt_is_exact_inverse(self, char):
        """Cifrar con C y descifrar con C† debe recuperar el statevector original."""
        amps, _ = char_to_amplitudes(char)
        sv_original = Statevector(amps)
        qc = build_qnn_cipher_circuit()

        sv_encrypted = sv_original.evolve(qc)
        sv_decrypted = sv_encrypted.evolve(qc.inverse())

        np.testing.assert_allclose(
            np.real(sv_decrypted.data), np.real(sv_original.data),
            atol=1e-9,
            err_msg=f"Statevector descifrado no coincide con el original para '{char}'"
        )


# ─── 4. Isomorfismo Nivel 3 (Flujo completo) ─────────────────────────────────

class TestLevel3Isomorphism:
    """
    Prueba de Oro: La ley de reversibilidad (Test del Tiempo).
    El sistema conservativo (circuito unitario CSWAP) debe ser
    perfectamente reversible: t → -t implica C† C = I.
    """

    @pytest.mark.parametrize("char", ASCII_CHARS)
    def test_full_pipeline_reversibility(self, char):
        """
        Ciclo completo: char → amps → Statevector → encrypt → decrypt → char_out.
        Valida el Isomorfismo Físico Nivel 3 del MANIFIESTO.
        """
        # Encode
        amps, bits = char_to_amplitudes(char)
        norm_factor = np.linalg.norm(bits)
        sv = Statevector(amps)

        # Encrypt (Feed-Forward)
        qc = build_qnn_cipher_circuit()
        sv_enc = sv.evolve(qc)

        # Decrypt (Feed-Backward)
        sv_dec = sv_enc.evolve(qc.inverse())

        # Decode
        char_out = amplitudes_to_char(np.real(sv_dec.data), norm_factor)

        assert char == char_out, (
            f"\n❌ Isomorfismo Nivel 3 FALLIDO para '{char}'\n"
            f"   Recuperado: '{char_out}'\n"
            f"   sv_original: {np.round(amps, 4)}\n"
            f"   sv_decrypted: {np.round(np.real(sv_dec.data), 4)}"
        )

    def test_encrypted_differs_from_original(self):
        """El statevector cifrado debe ser diferente al original (cifrado efectivo)."""
        amps, _ = char_to_amplitudes("A")
        sv = Statevector(amps)
        qc = build_qnn_cipher_circuit()
        sv_enc = sv.evolve(qc)
        # No deben ser idénticos (el cifrado debe mezclar fases)
        assert not np.allclose(np.real(sv_enc.data), np.real(sv.data), atol=1e-9), \
            "El cifrado CSWAP no modificó el statevector — la encriptación no es efectiva"
