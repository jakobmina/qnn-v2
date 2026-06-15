"""
h7_qnn_hash.py
Prueba de Concepto (PoC): Autómata Cuántico Autosustentado.
Usa la covarianza como función de pérdida para actualizar el estado inicial 'n',
generando una trayectoria termodinámica/disipativa para un "Metriplectic Hash".
"""
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
from qiskit.primitives import StatevectorSampler
from h7_bridge import run_h7_bridge
from utf8_qnn_poc import string_to_qnn_seed

# === CONSTANTES ===
pi, phi = np.pi, (1 + math.sqrt(5)) / 2
simulator = AerSimulator()

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

def run_qnn_step(n, iteration):
    n_z7 = 7 if (n % 7 == 0 and n != 0) else (n % 7)
    c_nz7 = 7 - n_z7
    
    o_n, i_n = math.cos(pi * n), math.cos(pi * phi * n)
    particle_class_value = (o_n * i_n) + i_n
    d_s = n + c_nz7 - n_z7
    observable_value = math.cos(pi * phi * d_s)

    su2_original = get_su2_matrix(k=n_z7, angle_moment=particle_class_value, mode='dynamic')
    
    # Qiskit Circuit
    qc_final = QuantumCircuit(3, 3)
    qc_final.h([0, 1, 2])
    qc_final.rz(c_nz7, 0)
    qc_final.ry(n_z7, 0)
    qc_final.rx(i_n, 0)
    qc_final.cswap(0, 2, 1)
    qc_final.ccx(2, 1, 0)
    qc_final.ccx(1, 0, 2)
    qc_final.measure_all()

    sampler = StatevectorSampler()
    job = sampler.run([qc_final], shots=1024)
    result = job.result()
    counts = result[0].data.meas.get_counts()
    
    total_shots = sum(counts.values())
    probabilidades = {estado: c / total_shots for estado, c in counts.items()}

    E_q1, E_q2, E_q1q2 = 0.0, 0.0, 0.0
    for estado, prob in probabilidades.items():
        q2, q1, q0 = int(estado[0]), int(estado[1]), int(estado[2])
        if q1 == 1: E_q1 += prob
        if q2 == 1: E_q2 += prob
        if q1 == 1 and q2 == 1: E_q1q2 += prob

    covariance_q1q2 = E_q1q2 - (E_q1 * E_q2)
    
    return {
        "n_z7": n_z7,
        "c_nz7": c_nz7,
        "covariance": covariance_q1q2,
        "probs": probabilidades,
        "su2": su2_original
    }

def generate_metriplectic_hash(seed, iterations=3):
    learning_rate = 1000  # Multiplicador termodinámico
    current_n = seed
    
    history = []
    
    for i in range(iterations):
        step_data = run_qnn_step(current_n, i)
        cov = step_data["covariance"]
        
        delta_n = int(cov * learning_rate * phi) 
        next_n = abs(current_n + delta_n + step_data['c_nz7'])
        
        history.append({
            "iteration": i + 1,
            "input_n": current_n,
            "n_z7": step_data['n_z7'],
            "c_nz7": step_data['c_nz7'],
            "covariance": cov,
            "delta_n": delta_n,
            "next_n": next_n
        })
        
        current_n = next_n

    # Final Hash via Bridge C
    final_step = run_qnn_step(current_n, iterations)
    dummy_sv = np.array([1, 0]) 
    
    bridge_out = run_h7_bridge(
        n=current_n,
        su2_matrix=final_step["su2"],
        statevector=dummy_sv,
        probabilities=final_step["probs"],
        export_path=f"his-torial/hash_n{current_n}",
        export_format="json",
        extra_metrics={"final_covariance": final_step["covariance"]},
        is_char=False
    )
    
    ms = bridge_out["MetriplecticState"]
    to_ = bridge_out["TorsionObservables"]
    
    return {
        "seed": seed,
        "final_n": current_n,
        "history": history,
        "hash": {
            "psi": float(ms.psi),
            "energy": float(ms.energy),
            "torsion": float(to_.spatial_torsion)
        }
    }

if __name__ == "__main__":
    print("=== QNN Metriplectic Hash PoC ===")
    raw_input = input("Insert seed n value or character: ")
    try:
        seed_n = int(raw_input)
    except ValueError:
        seed_n = string_to_qnn_seed(raw_input)

    result = generate_metriplectic_hash(seed_n, iterations=3)
    
    for step in result["history"]:
        print(f"\n--- Iteración {step['iteration']} ---")
        print(f"  Entrada n: {step['input_n']} (Z7={step['n_z7']})")
        print(f"  Covarianza resultante: {step['covariance']:.6f}")
        print(f"  [Loss/Update] Δn = {step['delta_n']} -> Siguiente n = {step['next_n']}")

    print("\n" + "="*40)
    print(f"[!] Autómata convergió en n_final = {result['final_n']}")
    print(f"\n==== METRIPLECTIC QUANTUM HASH ====")
    print(f"  Campo Psi: {result['hash']['psi']:.8f}")
    print(f"  Energía (H): {result['hash']['energy']:.8f}")
    print(f"  Torsión Espacial (S): {result['hash']['torsion']:.8f}")
    print("===================================\n")
