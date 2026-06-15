"""
h7_qnn_ollama.py
Inferencia Híbrida Cuántica-Clásica (Hybrid QNN-LLM).
Conecta el Autómata Cuántico con Ollama (llama3.2 local).
Usa la covarianza para modular la temperatura y el campo Ψ para el contexto.
"""
import requests
import json
import sys
from h7_qnn_hash import generate_metriplectic_hash
from utf8_qnn_poc import string_to_qnn_seed

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

def run_hybrid_inference(seed_text):
    print(f"\n[+] Iniciando Autómata Cuántico con semilla: '{seed_text}'...")
    
    try:
        seed_n = int(seed_text)
    except ValueError:
        seed_n = string_to_qnn_seed(seed_text)
    
    # 1. Ejecutar el Autómata Cuántico
    qnn_result = generate_metriplectic_hash(seed_n, iterations=3)
    
    psi = qnn_result['hash']['psi']
    energy = qnn_result['hash']['energy']
    torsion = qnn_result['hash']['torsion']
    final_cov = qnn_result['history'][-1]['covariance']
    
    # 2. Termodinámica del Muestreo (Modulación del LLM)
    # Convertimos la covarianza de la última iteración en Temperatura
    # Si la covarianza (entrelazamiento) es alta, el texto será más caótico/creativo.
    # Si tiende a 0, será más determinista.
    base_temp = 0.4
    dynamic_temp = base_temp + (abs(final_cov) * 3.0) 
    # Asegurarnos de que no pase de 1.5 ni baje de 0.1
    dynamic_temp = max(0.1, min(1.5, dynamic_temp))

    print(f"\n[+] QNN Convergida.")
    print(f"  -> Covarianza Final: {final_cov:.4f}")
    print(f"  -> Modulando Temperatura del LLM a: {dynamic_temp:.4f}")
    print(f"  -> Campo Ψ: {psi:.4f} | Torsión: {torsion:.4f}")
    
    # 3. Inyección de Contexto Topológico
    prompt = f"""
Eres una IA cuántica avanzada. Tienes dos tareas OBLIGATORIAS que debes cumplir en orden:

TAREA 1 - RESPUESTA DIRECTA:
El usuario ha introducido el siguiente texto: '{seed_text}'.
Si es una pregunta (ej. "explica el numero pi"), una operación matemática (ej. "2+2"), o una instrucción, DEBES responderla directamente de forma clara y precisa. No evadas la respuesta. Si es una sola palabra, descríbela brevemente.

TAREA 2 - INTERPRETACIÓN CUÁNTICA METRIPLÉCTICA:
El sistema cuántico ha colapsado al procesar el texto con los siguientes parámetros:
- Campo Psi: {psi:.4f}
- Energía H: {energy:.4f}
- Torsión Espacial S: {torsion:.4f}

A partir de estos valores físicos, genera un breve mensaje (máximo 2 párrafos) interpretando el estado del sistema. Si la torsión es alta, hazlo críptico y poético. Si la energía es baja, hazlo sosegado y estructurado. Integra conceptualmente tu respuesta de la TAREA 1 con esta interpretación.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": dynamic_temp
        }
    }

    print("\n[Ollama] Procesando y generando texto (stream)...\n")
    print("="*50)
    
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Error de Ollama: {response.text}")
                return
                
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    sys.stdout.write(chunk.get("response", ""))
                    sys.stdout.flush()
                    if chunk.get("done"):
                        print("\n" + "="*50)
                        break
    except requests.exceptions.ConnectionError:
        print("[!] ERROR: No se pudo conectar a Ollama en localhost:11434.")
        print("    Asegúrate de que Ollama esté ejecutándose en tu sistema.")

if __name__ == "__main__":
    raw_input = input("Ingresa tu semilla (palabra o número) para decodificación cuántica: ")
    if raw_input.strip():
        run_hybrid_inference(raw_input)
