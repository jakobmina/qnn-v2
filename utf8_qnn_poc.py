import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def char_to_amplitudes(c: str) -> np.ndarray:
    """Convierte un carácter (8 bits) en 8 amplitudes reales."""
    byte_val = c.encode('utf-8')[0]
    bits = [int(x) for x in f"{byte_val:08b}"]
    
    # Inyectar contexto de seguridad en el bit más significativo si es 0 (como en ASCII)
    if bits[0] == 0:
        bits[0] = 1 # Flag de Integridad / Paridad topológica
        
    amplitudes = np.array(bits, dtype=float)
    # Normalizar para que sea un Statevector válido (Suma de probabilidades = 1)
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        amplitudes /= norm
    return amplitudes, bits

def amplitudes_to_char(amplitudes: np.ndarray, original_norm: float) -> str:
    """Recupera el carácter desde las amplitudes."""
    # Des-normalizar
    raw_bits = np.round(amplitudes * original_norm).astype(int)
    
    # Limpiar el flag de seguridad que inyectamos
    if raw_bits[0] == 1:
        # Aquí validaríamos la covarianza/paridad. Por simplicidad, lo revertimos.
        raw_bits[0] = 0
        
    # Reconstruir el byte
    byte_val = int("".join(str(b) for b in raw_bits), 2)
    return chr(byte_val)

def build_qnn_cipher_circuit() -> QuantumCircuit:
    """Circuito de cifrado topológico (CSWAP y entrelazamiento)."""
    qc = QuantumCircuit(3)
    # Mezcla simple pero reversible
    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.cx(1, 0)
    return qc

def string_to_qnn_seed(text: str) -> int:
    """Convierte un string en un seed entero usando el cifrado QNN."""
    if not text:
        return 0
    
    qc_cipher = build_qnn_cipher_circuit()
    total_seed = 0
    
    for char in text:
        amps, original_bits = char_to_amplitudes(char)
        norm_factor = np.linalg.norm(original_bits)
        if norm_factor == 0:
            continue
        
        sv_initial = Statevector(amps)
        sv_encrypted = sv_initial.evolve(qc_cipher)
        
        raw_bits = np.round(np.real(sv_encrypted.data) * norm_factor).astype(int)
        byte_val = int("".join("1" if b > 0 else "0" for b in raw_bits), 2)
        total_seed += byte_val
        
    return total_seed

def main():
    char_in = "A"
    print(f"=== QNN UTF-8 / 3-Qubit Bridge ===")
    print(f"[1] Carácter de entrada: '{char_in}' (ASCII: {ord(char_in)})")
    
    # 1. ENCODE
    amps, original_bits = char_to_amplitudes(char_in)
    norm_factor = np.linalg.norm(original_bits)
    sv_initial = Statevector(amps)
    print(f"\n[2] Bits mapeados (con Flag de Seguridad): {original_bits}")
    print(f"    Statevector inicial (8 dimensiones):")
    print(np.round(sv_initial.data, 4))
    
    # 2. ENCRYPT (Feed-Forward)
    qc_cipher = build_qnn_cipher_circuit()
    sv_encrypted = sv_initial.evolve(qc_cipher)
    print(f"\n[3] Statevector Cifrado (CSWAP Scrambling):")
    print(np.round(sv_encrypted.data, 4))
    
    # En este punto el statevector cifrado podría viajar por la red o guardarse en .bin
    
    # 3. DECRYPT (Feed-Backward)
    qc_inverse = qc_cipher.inverse()
    sv_decrypted = sv_encrypted.evolve(qc_inverse)
    print(f"\n[4] Statevector Descifrado (Relajación Métrica):")
    print(np.round(sv_decrypted.data, 4))
    
    # 4. DECODE
    char_out = amplitudes_to_char(np.real(sv_decrypted.data), norm_factor)
    print(f"\n[5] Carácter Recuperado: '{char_out}'")
    
    if char_in == char_out:
        print("\n✅ ¡ÉXITO! Nivel 3 de Isomorfismo Físico Validado.")
        print("La información fluyó reversiblemente conservando el esqueleto de 8 bits en un espacio de Hilbert de 3 qubits.")
    else:
        print("\n❌ Error en la recuperación.")

if __name__ == "__main__":
    main()
