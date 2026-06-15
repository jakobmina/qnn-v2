"""
h7_main_integrated.py
Script H7 original con bridge C integrado al final del pipeline.
Inserta llamada a run_h7_bridge() luego de calcular probabilidades.
"""
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, Statevector
from qiskit.primitives import StatevectorSampler
from h7_bridge import run_h7_bridge, h7_to_metriplectic_state  # ← bridge
from utf8_qnn_poc import char_to_amplitudes

# === CONSTANTES ===
pi, phi = np.pi, (1 + math.sqrt(5)) / 2

raw_input = input("Insert n value or character: ")
is_char = False
try:
    n = int(raw_input)
    char_val = None
except ValueError:
    char_val = raw_input[0] if raw_input else "A"
    n = ord(char_val)
    is_char = True

# === SIMULACIÓN AER ===
simulator = AerSimulator()
def get_sv(matrix):
    qc = QuantumCircuit(1)
    qc.unitary(Operator(matrix), [0])
    qc.save_statevector()
    return simulator.run(qc).result().get_statevector(qc).data
# === APLANADO TOPOLÓGICO Z7 ===
n_z7 = 7 if (n % 7 == 0 and n != 0) else (n % 7)
is_vacuum = (math.gcd(n, 7) == 7)
# === VALORES QASIPERIODICOS
c_nz7 = 7 - n_z7
o_n, i_n = math.cos(pi * n), math.cos(pi * phi * n)
particle_class_value = (o_n * i_n) + i_n
particle_type = "fermionic" if np.isclose(particle_class_value, 0.0, atol=1e-9) else "bosonic"
d_s = n + c_nz7 - n_z7
observable_value = math.cos(pi * phi * d_s)

# === SU(2) ===
def get_su2_matrix(k, angle_moment, mode='dynamic'):
    if mode == 'dynamic':
        theta_rot = angle_moment * (pi / 4)
        axis = np.array([math.sin(k), math.cos(k), math.sin(k * phi)])
        norm = np.linalg.norm(axis)
        axis = axis / norm if norm > 0 else np.array([0, 0, 1])
        w, [x, y, z] = math.cos(theta_rot / 2), math.sin(theta_rot / 2) * axis
    else:
        half = (2 * pi * k / 7) / 2
        w, x, y, z = math.cos(half), 0.0, 0.0, math.sin(half)
    return np.array([[w + 1j*z, y + 1j*x], [-y + 1j*x, w - 1j*z]], dtype=complex)

su2_original  = get_su2_matrix(k=n_z7, angle_moment=particle_class_value, mode='dynamic')
su2_complement = get_su2_matrix(k=c_nz7, angle_moment=observable_value, mode='dynamic')

sv_orig = np.round(get_sv(su2_original), 4)
sv_comp = np.round(get_sv(su2_complement), 4)

contra_binario = f"{c_nz7:03b}"
estado_qubit   = f"{n_z7:03b}"

observable_value = math.cos(pi * phi * d_s)
# === REPORTE DE INTERFAZ H7 ===
print(f"\n--- H7 Particle Classification ---\nn original: {n} → Z7 flattened: {n_z7}\n  qubit_state (binary): {estado_qubit}\n  Statevector (original): {sv_orig}\nVacuum/boundary state: {is_vacuum}")
print(f"\nFull nodes operatives (n={n}):\n  parity: {o_n:.6f}\n  quasiperiod: {i_n:.6f}\n  chiral: {o_n*i_n+i_n:.6f}")
print(f"\nQuantum observables & SU(2):\n  quasiperiod_moment: {particle_class_value:.6f}\nentrelazamiento entre estado observado y oculto\n  observable_index: {n_z7}, {estado_qubit}")
print(f" Hidden index  n: {d_s}, hidden index: {c_nz7}, mapping:{contra_binario}\n  Statevector (original): {sv_orig}\n  Statevector (complementario): {sv_comp}")
print(f"\nClassification:\n  Particle type: {particle_type}\n  Quark composition: {estado_qubit}\n  Quaternionic axis: Z₇[{n_z7}] × Q₈[{estado_qubit}]")
print(f"  → {'Pauli exclusion (topological, neutron)' if particle_type == 'fermionic' else 'Continuous flow (JIT queue, proton)'}\n" + "----"*10)

# === GENERACIÓN DE CIRCUITOS (OPENQASM & QISKIT) ===
print(f"=== OpenQASM Gen (3-Qubit Equivalent Mapping) ===\n// Inicialización\nh q[0]; h q[1]; h q[2];\n// Modulación por fase\nrz({n:.4f}) q[0];\nry({c_nz7:.4f}) q[0];\nrx({i_n}) q[0];\ncswap q[0], q[2], q[1];\nccx q[2], q[1], q[0];\nccx q[1], q[0], q[2];\nmeasure q -> c;")

print(f"\n=== Qiskit Circuit ===\n{sv_comp}")
qc_final = QuantumCircuit(3, 3)

if is_char:
    byte_val = char_val.encode('utf-8')[0]
    bits = [int(x) for x in f"{byte_val:08b}"]
    if bits[0] == 0: bits[0] = 1 # Flag seguridad
    amps = np.array(bits, dtype=float)
    norm = np.linalg.norm(amps)
    if norm > 0: amps /= norm
    
    qc_final.initialize(amps, [0, 1, 2])
    
    sv_encrypted = Statevector.from_instruction(qc_final.copy().remove_final_measurements(inplace=False))
    sv_for_export = sv_encrypted.data
else:
    qc_final.h([0, 1, 2])
    qc_final.rz(c_nz7, 0)
    qc_final.ry(n_z7, 0)
    qc_final.rx(i_n, 0)
    qc_final.cswap(0, 2, 1)
    qc_final.ccx(2, 1, 0)
    qc_final.ccx(1, 0, 2)
    sv_for_export = sv_orig # Mantener el statevector original 1-qubit para n numérico

# Move measure_all() before running the sampler to get classical counts
qc_final.measure_all()

# imprimir probabilidades
statevector_from_su2 = Operator(su2_original)
sampler = StatevectorSampler()
job = sampler.run([qc_final], shots=1024) # Se pasa como una lista [pub]
result = job.result()

# 3. Extraer los conteos usando el nuevo formato indexado por registros
data_pub = result[0].data
counts = data_pub.meas.get_counts() # Acceso directo al registro clásico "meas"

# 4. Convertir a probabilidades relativas
total_shots = sum(counts.values())
probabilidades = {estado: c / total_shots for estado, c in counts.items()}

print("Probabilidades en Qiskit v2.x:", probabilidades)
print(qc_final.draw())

# 5. Extraer Covarianza y Asimetría
E_q1, E_q2, E_q1q2 = 0.0, 0.0, 0.0
P_10, P_01 = 0.0, 0.0

for estado, prob in probabilidades.items():
    # En Qiskit el string es 'q2 q1 q0'
    q2, q1, q0 = int(estado[0]), int(estado[1]), int(estado[2])
    if q1 == 1: E_q1 += prob
    if q2 == 1: E_q2 += prob
    if q1 == 1 and q2 == 1: E_q1q2 += prob
    if q2 == 1 and q1 == 0: P_10 += prob
    if q2 == 0 and q1 == 1: P_01 += prob

covariance_q1q2 = E_q1q2 - (E_q1 * E_q2)
asymmetry_q1q2 = P_10 - P_01

print(f"\n--- Covarianza y Asimetría del Circuito (CSWAP) ---")
print(f"  E[q1]: {E_q1:.6f}, E[q2]: {E_q2:.6f}")
print(f"  Cov(q1, q2): {covariance_q1q2:.6f}")
print(f"  Asimetría P(q2=1,q1=0) - P(q2=0,q1=1): {asymmetry_q1q2:.6f}")

export_name = f"h7_state_char_{char_val}" if is_char else f"h7_state_n{n}"
export_path = f"his-torial/{export_name}"

# ── BRIDGE C (nueva integración) ──────────────────────────────────
bridge_out = run_h7_bridge(
    n            = n,
    su2_matrix   = su2_original,
    statevector  = sv_for_export,
    probabilities= probabilidades,
    export_path  = export_path,
    export_format= "both",       # genera .json y .bin
    extra_metrics= {
        "covariance_q1q2": covariance_q1q2,
        "asymmetry_q1q2": asymmetry_q1q2
    },
    is_char      = is_char
)

ms   = bridge_out["MetriplecticState"]
to_  = bridge_out["TorsionObservables"]
ec   = bridge_out["EstadoCuantico"]
grid = bridge_out["QNNGrid"]

print(f"\n[Bridge C] MetriplecticState → psi={ms.psi:.6f}, energy={ms.energy:.6f}")
print(f"[Bridge C] TorsionObservables → density={to_.energy_density:.6f}, "
      f"torsion={to_.spatial_torsion:.6f} (DRIFT_072)")
print(f"[Bridge C] QNNGrid layers[0] → weight={grid.layers[0].weight:.6f}, "
      f"pair=({n_z7}, {c_nz7})")
print(f"[Bridge C] Exportado: {export_name}.json + {export_name}.bin")
