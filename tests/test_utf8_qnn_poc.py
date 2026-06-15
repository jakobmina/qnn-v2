import numpy as np
import pytest
from utf8_qnn_poc import char_to_amplitudes, amplitudes_to_char, build_qnn_cipher_circuit, string_to_qnn_seed

def test_char_to_amplitudes_and_back():
    char_in = "X"
    amps, original_bits = char_to_amplitudes(char_in)
    norm_factor = np.linalg.norm(original_bits)
    
    char_out = amplitudes_to_char(amps, norm_factor)
    assert char_in == char_out

def test_string_to_qnn_seed():
    seed1 = string_to_qnn_seed("test")
    seed2 = string_to_qnn_seed("test")
    seed3 = string_to_qnn_seed("other")
    
    assert isinstance(seed1, int)
    assert seed1 == seed2 # Debe ser determinista para la misma entrada
    assert seed1 != seed3 # Diferente entrada debe generar diferente seed

def test_string_to_qnn_seed_empty():
    assert string_to_qnn_seed("") == 0
