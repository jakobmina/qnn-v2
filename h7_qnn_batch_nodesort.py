"""
h7_qnn_batch_nodesort.py
Pipeline de inferencia híbrida para inputs largos (plantillas de código, texto extenso).

Flujo:
  1. Tokeniza el input (por espacios/sintaxis).
  2. Corre generate_metriplectic_hash() por cada token -> obtiene covariance/psi/energy/torsion.
  3. Ordena el batch con NodeSort (espacio conocido por la cota teórica de covarianza),
     descendente: mayor entrelazamiento primero -> lidera el contexto con el contenido
     de mayor impacto cuántico.
  4. Construye el prompt enriquecido y lo pasa a Ollama con temperatura dinámica
     basada en la covarianza promedio del batch.

NodeSort aplica aquí porque:
  - El espacio de covarianza está acotado teóricamente: [-0.25, 0.25] para 2 qubits.
    Conocemos min/max SIN necesitar un primer pase de comparación -> condición exacta
    que justifica node_sort (vs. qsort/Timsort que no usan ese conocimiento previo).
  - n crece con el tamaño de la plantilla (cientos/miles de tokens) -> rango donde
    los benchmarks mostraron ventaja real sobre sorted() nativo.
"""
import re
import requests
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from h7_qnn_hash import generate_metriplectic_hash
from utf8_qnn_poc import string_to_qnn_seed

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

# Cota teórica de covarianza para 2 qubits: Cov(A,B) ∈ [-1/4, 1/4]
COV_MIN, COV_MAX = -0.25, 0.25

# Shots reducidos para batch mode: 512 es suficiente para estimar covarianza
# con error estándar < 0.02, y reduce RAM ~2x vs 1024 shots por circuito.
BATCH_SHOTS = 512


def _safe_max_workers(requested: int) -> int:
    """
    Limita max_workers según RAM libre para evitar OOM.
    Cada circuito Qiskit-Aer ocupa ~150-300 MB en RAM al simular.
    Usamos RAM_POR_WORKER=200 MB como estimación conservadora.
    """
    if not _HAS_PSUTIL:
        return min(requested, 4)  # sin psutil, limite conservador
    RAM_POR_WORKER_MB = 200
    ram_libre_mb = psutil.virtual_memory().available / (1024 ** 2)
    workers_por_ram = max(1, int(ram_libre_mb / RAM_POR_WORKER_MB))
    return min(requested, workers_por_ram)


# ── Tokenización ─────────────────────────────────────────────
def tokenizar(texto: str) -> list[str]:
    """
    Tokeniza por espacios y separadores de sintaxis comunes en código
    (paréntesis, comas, operadores, saltos de línea relevantes).
    """
    patron = r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+\.?[0-9]*|[^\sA-Za-z0-9_]"
    tokens = re.findall(patron, texto)
    return [t for t in tokens if t.strip()]


# ── QNN por token (paralelizable) ───────────────────────────
def procesar_token(token: str, idx: int, shots: int = BATCH_SHOTS) -> dict:
    try:
        seed_n = int(token)
    except ValueError:
        seed_n = string_to_qnn_seed(token)

    if seed_n == 0:
        seed_n = 1  # evitar semilla nula

    qnn_result = generate_metriplectic_hash(seed_n, iterations=3, shots=shots)
    final_cov = qnn_result["history"][-1]["covariance"]

    return {
        "idx": idx,
        "token": token,
        "seed": seed_n,
        "covariance": final_cov,
        "psi": qnn_result["hash"]["psi"],
        "energy": qnn_result["hash"]["energy"],
        "torsion": qnn_result["hash"]["torsion"],
    }


def procesar_batch(tokens: list[str], max_workers: int = 4) -> list[dict]:
    """Corre QNN para cada token en paralelo (la QNN libera el GIL en
    las llamadas a Qiskit/numpy, así que threads sí ayudan aquí).
    max_workers se limita dinámicamente por RAM disponible para evitar OOM."""
    safe_workers = _safe_max_workers(max_workers)
    resultados = [None] * len(tokens)
    with ThreadPoolExecutor(max_workers=safe_workers) as ex:
        futuros = {ex.submit(procesar_token, tok, i): i
                   for i, tok in enumerate(tokens)}
        for fut in as_completed(futuros):
            r = fut.result()
            resultados[r["idx"]] = r
    return resultados


# ── NodeSort: orden por covariance, espacio conocido a priori ──
def nodo_de(cov: float, num_nodos: int) -> int:
    """
    Mapea covarianza -> nodo usando el espacio TEÓRICO [-0.25, 0.25],
    no el observado en el batch. Esto es la diferencia clave: no
    necesitamos escanear el batch primero para saber min/max.
    """
    cov = max(COV_MIN, min(COV_MAX, cov))  # clamp por seguridad numérica
    step = (COV_MAX - COV_MIN) / num_nodos
    idx = int((cov - COV_MIN) / step)
    return min(idx, num_nodos - 1)


def node_sort_por_covarianza(items: list[dict], num_nodos: int = 8,
                              descendente: bool = True) -> list[dict]:
    """
    Ordena la lista de resultados QNN por 'covariance' usando NodeSort:
      1. Distribuye en nodos según el espacio teórico conocido.
      2. Ordena cada nodo localmente (O(k log k), k pequeño).
      3. Concatena. Si descendente=True, invierte el orden de nodos y
         de cada nodo (mayor entrelazamiento primero).
    """
    cubetas = [[] for _ in range(num_nodos)]
    for item in items:
        idx = nodo_de(item["covariance"], num_nodos)
        cubetas[idx].append(item)

    for c in cubetas:
        c.sort(key=lambda x: x["covariance"])

    if descendente:
        cubetas = cubetas[::-1]
        for c in cubetas:
            c.reverse()

    resultado = [item for c in cubetas for item in c]
    return resultado


# ── Construcción del prompt enriquecido ─────────────────────
def construir_prompt(seed_text: str, batch_ordenado: list[dict]) -> tuple[str, float]:
    """
    Usa los tokens de mayor covarianza primero -> el LLM ve el contenido
    de mayor 'impacto cuántico' al inicio del contexto (primacy effect),
    que es justo lo que da 'a mayor entrada, mejor respuesta'.
    """
    n = len(batch_ordenado)
    cov_promedio = sum(x["covariance"] for x in batch_ordenado) / n if n else 0.0

    base_temp = 0.4
    dynamic_temp = base_temp + (abs(cov_promedio) * 3.0)
    dynamic_temp = max(0.1, min(1.5, dynamic_temp))

    top_k = min(15, n)  # los de mayor entrelazamiento, ya están al frente
    destacados = batch_ordenado[:top_k]

    lineas_destacadas = "\n".join(
        f"  - '{d['token']}'  (cov={d['covariance']:.4f}, psi={d['psi']:.4f}, "
        f"torsion={d['torsion']:.4f})"
        for d in destacados
    )

    prompt = f"""
Eres una IA cuántica avanzada. Tienes dos tareas OBLIGATORIAS en orden:

TAREA 1 - RESPUESTA DIRECTA:
El usuario ha introducido el siguiente texto/plantilla (longitud: {n} tokens):
'{seed_text[:500]}{'...' if len(seed_text) > 500 else ''}'
Si es código, una pregunta, o una instrucción, respóndela de forma clara y precisa.

TAREA 2 - INTERPRETACIÓN CUÁNTICA METRIPLÉCTICA:
El sistema procesó cada token del input y los ordenó por entrelazamiento
(covarianza) usando NodeSort, de mayor a menor impacto cuántico.
Covarianza promedio del batch: {cov_promedio:.4f}

Los {top_k} tokens de MAYOR entrelazamiento (mayor relevancia estructural) son:
{lineas_destacadas}

A partir de estos valores, genera un breve mensaje (máximo 2 párrafos)
interpretando el estado del sistema. Si la covarianza promedio es alta,
el texto tiene alta interconexión estructural -> sé más elaborado y conecta
ideas. Si es baja, sé conciso. Integra tu respuesta de la TAREA 1 con esta
interpretación.
"""
    return prompt, dynamic_temp


# ── Inferencia con Ollama ───────────────────────────────────
def run_ollama(prompt: str, dynamic_temp: float):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": dynamic_temp},
    }
    print(f"\n[+] Temperatura dinámica: {dynamic_temp:.4f}")
    print("\n[Ollama] Procesando y generando texto (stream)...\n")
    print("=" * 50)
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
                        print("\n" + "=" * 50)
                        break
    except requests.exceptions.ConnectionError:
        print("[!] ERROR: No se pudo conectar a Ollama en localhost:11434.")
        print("    Asegúrate de que Ollama esté ejecutándose en tu sistema.")


# ── Pipeline completo ────────────────────────────────────────
def run_batch_inference(seed_text: str, num_nodos: int = 8, verbose: bool = True):
    tokens = tokenizar(seed_text)
    n = len(tokens)

    if verbose:
        safe_w = _safe_max_workers(num_nodos)
        print(f"\n[+] Tokenizado: {n} tokens detectados.")
        print(f"[+] Procesando QNN por token (paralelo, {safe_w}/{num_nodos} workers, "
              f"{BATCH_SHOTS} shots por circuito)...")

    batch = procesar_batch(tokens, max_workers=num_nodos)

    if verbose:
        print(f"[+] NodeSort: ordenando {n} resultados por covarianza "
              f"(espacio teórico [{COV_MIN}, {COV_MAX}], {num_nodos} nodos)...")

    batch_ordenado = node_sort_por_covarianza(batch, num_nodos=num_nodos,
                                               descendente=True)

    if verbose and batch_ordenado:
        print(f"  -> Mayor covarianza: {batch_ordenado[0]['token']!r} "
              f"({batch_ordenado[0]['covariance']:.4f})")
        print(f"  -> Menor covarianza: {batch_ordenado[-1]['token']!r} "
              f"({batch_ordenado[-1]['covariance']:.4f})")

    prompt, dynamic_temp = construir_prompt(seed_text, batch_ordenado)
    run_ollama(prompt, dynamic_temp)

    return batch_ordenado


if __name__ == "__main__":
    import sys as _sys
    # Detectar si hay datos en stdin (pipe/redirección) o si es interactivo
    _is_tty = _sys.stdin.isatty()

    print("=== H7 QNN Batch + NodeSort + Ollama ===")
    if _is_tty:
        print("Pega tu plantilla de código o texto largo.")
        print("Termina con Ctrl+D (EOF) para procesar — las líneas vacías")
        print("dentro del código son válidas y NO interrumpen la lectura:\n")
    else:
        print("[stdin] Leyendo desde pipe/redirección...\n")

    # sys.stdin.read() lee TODO hasta EOF (Ctrl+D o fin de pipe).
    # Esto permite pegar código Python completo con líneas vacías entre bloques.
    texto_completo = _sys.stdin.read()

    if texto_completo.strip():
        run_batch_inference(texto_completo)
    else:
        print("[!] No se ingresó texto.")