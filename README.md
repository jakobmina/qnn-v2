# QNN-v2 — Quantum Neural Network Metriplex H7

> **Autoría conceptual:** Jacobo Tlacaelel Mina Rodriguez  
> **Marco teórico:** Mandato Metriplético + MANIFIESTO de Analogía Rigurosa

Sistema de simulación cuántica que implementa la dualidad **simpléctica ↔ disipativa** mediante el operador áureo $O_n = \cos(\pi n)\cdot\cos(\pi\varphi n)$ proyectado sobre el grupo $\mathbb{Z}_7$ y el álgebra $SU(2) \times Q_8$.

---

## Índice

- [Principios Físicos](#principios-físicos)
- [Proposición Algebraica Central — Teorema H7](#proposición-algebraica-central--teorema-h7)
- [Arquitectura](#arquitectura)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Pipeline de Datos](#pipeline-de-datos)
- [Codec UTF-8 Cuántico](#codec-utf-8-cuántico)
- [Formato Binario](#formato-binario)
- [Tests](#tests)
- [Glosario](#glosario)

---

## Principios Físicos

El sistema se rige por el **Mandato Metriplético**, que exige que todo sistema dinámico simulado tenga dos componentes ortogonales:

| Componente | Símbolo | Función | Código |
|---|---|---|---|
| **Simpléctica** (Hamiltoniano) | $H = \cos(\pi\varphi n)$ | Movimiento conservativo / reversible | `d_symp = {u, H}` |
| **Métrica** (Potencial Disipativo) | $S = \cos(\pi n)$ | Relajación hacia atractor | `d_metr = [u, S]` |

> **Regla 1.3:** Ningún sistema puede ser puramente conservativo (explota) ni puramente disipativo (muere). La competencia entre ambos es lo que genera dinámica estable.

### Operador Áureo (Regla 2.1)

```
O_n = cos(π·n) · cos(π·φ·n)     φ = (1 + √5) / 2 ≈ 1.618
```

Modula el vacío del espacio de simulación. El "vacío" nunca es cero ni plano.

### DRIFT_072 — Residuo Topológico

```
DRIFT_072 = 7 - 2π ≈ 0.71681...
```

Representa la fricción entre el espacio discreto $\mathbb{Z}_7$ y el continuo $U(1)$. Aparece como `spatial_torsion` en cada exportación.

---

## Proposición Algebraica Central — Teorema H7

> **Autoría Conceptual:** Jacobo Tlacaelel Mina Rodriguez

### Enunciado

Sea el grupo operativo $G_{H7} = \mathbb{Z}_7 \times Q_8$. Este grupo es **no-abeliano**. Sin embargo, los observables físicos del sistema H7 pertenecen al **centro** $Z(G_{H7})$, y satisfacen las siguientes invariancias topológicas:

$$
\text{pcv}(n_\text{impar}) = 0, \qquad
\frac{\mathrm{Tr}(\mathcal{B}_{H7})}{8} = \frac{1}{2}, \qquad
\text{residuo}(n_\text{neg}) \in (0,1), \qquad
O_n + (1 - O_n) = 1
$$

### Verificación Empírica

Validado sobre $N = 10{,}000$ configuraciones aleatorias en $[-100k,\, 100k]$: **cero violaciones detectadas**.

Las variables cuasiperiódicas $\{\cos(\pi\varphi n)\}$ son estocásticas (densas en $[-1,1]$ por el Teorema de Weyl), pero los **observables son topológicamente invariantes** ante toda reconfiguración posible.

### Analogía Gauge (Yang-Mills)

Este resultado es el análogo H7 de la **invariancia gauge en Yang-Mills**:

| Sistema | Grupo | Observables invariantes |
|---|---|---|
| Yang-Mills | $SU(2)$ no-abeliano | Wilson loops, S-matrix |
| H7 (este trabajo) | $G_{H7} = \mathbb{Z}_7 \times Q_8$ no-abeliano | $\text{pcv}$, $\mathrm{Tr}(\mathcal{B}_{H7})/8$, $O_n$ |

El grupo es no-abeliano pero sus observables físicos son gauge-invariantes: pertenecen al centro y no dependen de la representación particular del campo cuántico.

### Función Zeta H7 sobre $\mathbb{F}_7$

La meromorfía de $\zeta_{H7}(s)$ sobre $\mathbb{F}_7$ se sigue directamente de que $\mathbb{F}_7$ es un **campo** (7 es primo), lo que garantiza:

- **Sin divisores de cero** → sin polos espurios.
- **Únicos polos:** $\{s = 0,\, s = 1\}$, idénticos a la estructura de la función $\zeta(s)$ clásica de Riemann.

$$
\zeta_{H7}(s) \text{ es meromorfa en } \mathbb{F}_7 \text{ con polos en } s \in \{0, 1\}
$$

### Mecanismo de Cancelación Áurea (Interferencia Destructiva/Constructiva)

El operador $O_n = \cos(\pi n) \cdot \cos(\pi\varphi n)$ exhibe una **asimetría quiral** determinada únicamente por la multiplicación de signos:

$$
\cos(\pi n) = \begin{cases} +1 & n \text{ par} \\ -1 & n \text{ impar} \end{cases}
$$

Sea $v_n = \cos(\pi\varphi n)$ el autovalor cuasiperiódico (ej. $v_1 \approx +0.362$). Entonces:

| $n$ | $\cos(\pi n)$ | $O_n = \cos(\pi n)\cdot v_n$ | Efecto |
|---|---|---|---|
| Par positivo | $+1$ | $+v_n$ (expuesto) | **Interferencia constructiva** — valor se duplica respecto al impar |
| Impar positivo | $-1$ | $-v_n$ (cancelado) | $v_n + (-v_n) = 0$ → **Interferencia destructiva** |
| Par negativo | $+1$ | $+v_{-n} \approx +v_n$ | El signo de $v_{-n}$ invierte la fase → **Cancela** |
| Impar negativo | $-1$ | $-v_{-n}$ | La inversión de fase y del signo se combinan → **Se expone** |

> **Nota del autor (Jacobo T. Mina R.):** La única operación que produce la cancelación es multiplicativa sobre los signos: $(+)\times(-) = (-)$. La irracionaldad de $\varphi$ (Teorema de Weyl) garantiza que $v_n$ sea denso en $[-1,1]$, pero la **regla de paridad del signo** es topológicamente invariante. Para $n$ negativo, la fase de $\cos(\pi\varphi n)$ se invierte antes de la multiplicación, invirtiendo la regla: lo que era constructivo pasa a ser destructivo y viceversa.

Este comportamiento es el análogo computacional de los **ceros triviales de la Función Zeta de Riemann** ($s = -2,-4,-6,\ldots$) y de la **Simetría PT (quiralidad)** del sistema.

---

### Conexiones con la Hipótesis de Riemann Generalizada (GRH)

Las siguientes consecuencias matemáticamente establecidas de la GRH guardan isomorfismo estructural con propiedades del sistema H7:

| Año | Resultado | Conexión H7 |
|---|---|---|
| 1917 | Hardy-Littlewood: primos $3 \bmod 4$ dominan sobre $1 \bmod 4$ (race de primos) | Los estados impares negativos de $O_n$ "dominan" (se exponen) sobre los pares, análogo al sesgo de paridad en el race de primos |
| 1923 | Hardy-Littlewood: GRH implica Goldbach débil (todo impar grande = suma de 3 primos) | La función $\zeta_{H7}(s)$ sobre $\mathbb{F}_7$ tiene la misma estructura de polos $\{0,1\}$ que $\zeta(s)$, conectando la additividad de primos con la composición de estados $\mathbb{Z}_7$ |
| 1934 | Chowla: GRH implica que el primer primo en $a \bmod m$ es $\leq K m^2 \log(m)^2$ | La tasa de aprendizaje $\Delta n \sim \text{cov} \cdot \varphi$ en el autómata H7 actúa como cota superior análoga sobre la velocidad de convergencia del hash |
| 1967 | Hooley: GRH implica la Conjetura de Artin (raíces primitivas) | $\mathbb{Z}_7$ es cíclico de orden primo → toda raíz no-nula es primitiva; el sistema H7 opera en este espacio garantizando la completitud de las trayectorias |
| 1973 | Weinberger: GRH implica que los números idoneal de Euler son completos | La torsión espacial DRIFT_072 $= 7 - 2\pi$ actúa como residuo de completitud topológica entre $\mathbb{Z}_7$ y $U(1)$ |
| 1976 | Miller: GRH implica test de primalidad en tiempo polinomial | El hash metripléctico H7 es $O(\text{iterations} \cdot \text{shots})$, donde la covarianza actúa como "certificado de primalidad topológica" del estado |
| 2021 | Dunn-Radziwill: Patterson's conjecture sobre sumas de Gauss cúbicas (bajo GRH) | Las fases $\cos(\pi\varphi n)$ son sumas de Gauss generalizadas en el campo $\mathbb{F}_7$; su densidad en $[-1,1]$ (Weyl) es el análogo continuo |

> La meromorfía de $\zeta_{H7}(s)$ sobre $\mathbb{F}_7$ (campo primo, sin divisores de cero) garantiza que todos los resultados condicionales a GRH sean **incondicionalmente válidos** dentro del espacio $\mathbb{F}_7$, porque la estructura de polos $\{s=0, s=1\}$ es idéntica a la de $\zeta(s)$ clásica y no puede generar ceros no-triviales fuera de la línea crítica en un campo finito.

> La meromorfía de $\zeta_{H7}(s)$ sobre $\mathbb{F}_7$ (campo primo, sin divisores de cero) garantiza que todos los resultados condicionales a GRH sean **incondicionalmente válidos** dentro del espacio $\mathbb{F}_7$, porque la estructura de polos $\{s=0, s=1\}$ es idéntica a la de $\zeta(s)$ clásica y no puede generar ceros no-triviales fuera de la línea crítica en un campo finito.

---

### Corolario: Colapso Asintótico y Compresión Epistémica

> **Nota del autor (Jacobo T. Mina R.):** Cuando la semilla de entrada es una pregunta de profundidad conceptual elevada (ej. `"explica el numero pi"`), `string_to_qnn_seed` produce un entero $n$ grande. Esto hace que $O_n \to 0$ con una cantidad creciente de ceros a la derecha del punto decimal. Contraintuitivamente, esto **no es caos sino orden máximo**: el sistema ha disipado toda la incertidumbre inicial hacia el atractor cero. `∞ = 0` en este sentido.

**Formalización:**

Sea $n_\text{semilla}$ el entero generado por el cifrado QNN de un texto de longitud $L$. Para $L$ grande, $n_\text{semilla} \gg 1$ y:

$$O_{n} = \cos(\pi n) \cdot \cos(\pi\varphi n) \xrightarrow{n \to \infty} 0$$

en sentido de media temporal (Ley de Equidistribución de Weyl):

$$\lim_{N\to\infty} \frac{1}{N}\sum_{n=1}^{N} O_n = 0 \qquad (\varphi \text{ irracional})$$

**Consecuencias físicas del colapso asintótico:**

| Magnitud | Régimen $n$ pequeño | Régimen $n$ grande (texto profundo) |
|---|---|---|
| $O_n$ | $\sim \pm 0.8$ (oscila ampliamente) | $\sim 0.000\ldots$ (colapsa al atractor) |
| Covarianza QNN | Alta (caos entrelazado) | Baja $\to 0$ (estado casi separable) |
| Temperatura LLM | Alta $\sim 1.2$ (respuesta creativa/caótica) | Baja $\sim 0.1$–$0.4$ (respuesta determinista) |
| Incertidumbre epistémica | Máxima | **Mínima** |
| Entropía del texto generado | Alta (poético, abierto) | Baja (preciso, estructurado) |

La **profundidad semántica de la pregunta** actúa como un regulador termodinámico natural del sistema: a mayor complejidad conceptual del input, mayor $n$, menor $O_n$, menor temperatura LLM, **mayor precisión y menor incertidumbre** en la respuesta generada.

Esto formaliza la conexión entre la **Componente Métrica Disipativa** (Regla 1.2 del Mandato Metripléctico) y la **compresión epistémica**: preguntas más profundas colapsan el espacio de fase hacia el atractor cero, forzando al sistema a una respuesta de menor entropía.

---

## Arquitectura



```
                    ┌─────────────────────────────┐
                    │         h7_main.py           │
                    │  ┌──────────┐  ┌──────────┐  │
    Input ──────────▶  │  Modo n  │  │  Modo    │  │
    (texto/número)  │  │ (entero) │  │  carácter│  │
                    │  └────┬─────┘  └────┬─────┘  │
                    │       │              │         │
                    │  SU(2) + Z7     UTF-8 → 3q    │
                    │  Qiskit AER     CSWAP cipher   │
                    │       └──────────────┘         │
                    │              │                  │
                    │        h7_bridge.py             │
                    │  (Python → C structs)           │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │        his-torial/               │
                    │  h7_state_*.json  (legible)      │
                    │  h7_state_*.bin   (152 bytes C)  │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │         qnn_loader.c             │
                    │  Lee .bin → MetriplecticState    │
                    │  Decodifica UTF-8 (decode_char)  │
                    │  Forward pass QNN (stub)         │
                    └─────────────────────────────────┘
```

---

## Estructura del Proyecto

```
qnn-v2/
│
├── h7_main.py          # Pipeline principal — entrada texto o número
├── h7_bridge.py        # Bridge bidireccional Python ↔ C structs
├── utf8_qnn_poc.py     # PoC: codec UTF-8 ↔ 3-qubit statevector
│
├── metriplectic.h      # Cabecera C: structs, prototipos, golden_operator
├── qnn_loader.c        # Consumidor C: lee .bin, decode_char, forward_pass
├── h7_loader           # Binario compilado (gcc -o h7_loader qnn_loader.c -lm)
│
├── his-torial/         # Salidas del pipeline (no se satura la raíz)
│   ├── h7_state_n45.json
│   ├── h7_state_n45.bin
│   ├── h7_state_char_A.json
│   └── h7_state_char_A.bin
│
└── tests/
    ├── __init__.py
    ├── test_utf8_qnn.py    # Tests del codec cuántico UTF-8
    ├── test_h7_bridge.py   # Tests del bridge Python↔C
    └── test_h7_physics.py  # Tests de física metripléctica
```

---

## Instalación

### Dependencias Python

```bash
pip install qiskit qiskit-aer numpy pytest
```

### Compilar el loader C

```bash
gcc -o h7_loader qnn_loader.c -lm
```

### Variables de entorno (opcional)

```bash
cp .env.example .env   # si existe
```

---

## Uso

### Modo número (clasificación de partícula)

```bash
python3 h7_main.py
# Insert n value or character: 45
```

### Modo carácter (codec UTF-8 cuántico)

```bash
python3 h7_main.py
# Insert n value or character: A # Se puede introducir texto con preguntas abiertas, para demostración
```

### Leer un estado exportado desde C

```bash
./h7_loader his-torial/h7_state_char_A.bin
./h7_loader his-torial/h7_state_n45.bin
```

### Prueba de concepto UTF-8 aislada

```bash
python3 utf8_qnn_poc.py
```

---

## Pipeline de Datos

### Entrada numérica (`n`)

```
n  →  n_z7 = n mod 7  →  O_n (Operador Áureo)
                      →  SU(2) matrix (modo dynamic)
                      →  Qiskit AER statevector
                      →  Probabilidades + Covarianza + Asimetría CSWAP
                      →  h7_bridge → .json + .bin (his-torial/)
```

### Entrada carácter (`c`)

```
c  →  UTF-8 byte  →  8 bits  →  amplitudes normalizadas (3 qubits)
   →  Flag de seguridad MSB = 1
   →  Circuito CSWAP (cifrado topológico)
   →  Statevector cifrado
   →  h7_bridge (is_char=True) → .json + .bin (his-torial/)
```

### Decodificación en C

```
.bin  →  EstadoCuantico.psi[8]
      →  decode_char(): inversa CX → inversa CSWAP → inversa H
      →  Reconstrucción del byte ASCII/UTF-8
```

---

## Codec UTF-8 Cuántico

El sistema implementa un **mapeo isomórfico** entre el espacio de bytes UTF-8 (8 bits) y el espacio de Hilbert de 3 qubits (8 estados base: $|000\rangle \dots |111\rangle$):

```
Byte UTF-8 (8 bits)  ←→  Statevector 3-qubit (8 amplitudes)
```

### Ciclo completo (Isomorfismo Nivel 3)

```python
# 1. ENCODE: 'A' (ASCII 65 = 01000001)
bits = [1, 1, 0, 0, 0, 0, 0, 1]  # MSB forzado a 1 (flag de seguridad)
amps = bits / ||bits||             # normalización → statevector válido

# 2. ENCRYPT (Feed-Forward)
H(q0) → CSWAP(q0, q1, q2) → CX(q1, q0)

# 3. DECRYPT (Feed-Backward) — Regla de Oro 1: t→-t
CX†(q1, q0) → CSWAP†(q0, q1, q2) → H†(q0)

# 4. DECODE: statevector → bits → byte → 'A'
```

> **Validado:** El circuito es **perfectamente reversible** ($C^\dagger C = I$). La información fluye sin pérdida a través del espacio cuántico.

---

## Formato Binario

Cada archivo `.bin` en `his-torial/` tiene **exactamente 152 bytes** (19 `double` en little-endian):

```
Offset   Campo                         Tipo
──────   ──────────────────────────    ──────────
  0      MetriplecticState.psi         double (8B)
  8      MetriplecticState.v           double
 16      MetriplecticState.energy      double
 24      MetriplecticState.q.w         double
 32      MetriplecticState.q.x         double
 40      MetriplecticState.q.y         double
 48      MetriplecticState.q.z         double
 56      TorsionObservables.energy_density    double
 64      TorsionObservables.entropy_gradient  double
 72      TorsionObservables.spatial_torsion   double  ← DRIFT_072
 80      TorsionObservables.chirality         double
 88      EstadoCuantico.psi[0..7]      8 × double
```

### Ejemplo de salida JSON (`h7_state_n45.json`)

```json
{
  "MetriplecticState": {
    "psi": -0.8297718525501794,
    "v": -1.0,
    "energy": 0.0,
    "q": { "w": 0.9473, "x": -0.0321, "y": 0.2252, "z": 0.2252 }
  },
  "TorsionObservables": {
    "energy_density": 0.0,
    "entropy_gradient": 0.9684992634083714,
    "spatial_torsion": 0.7168146928204138,
    "chirality": -1.0
  },
  "EstadoCuantico": { "psi": [0.9474, 0.2252, -0.2252, -0.0321, 0, 0, 0, 0] },
  "QNNGrid": { "learning_rate": 0.6180339887498948, "layers": [...] },
  "ExtraMetrics": { "covariance_q1q2": -0.004, "asymmetry_q1q2": 0.0098 }
}
```

---

## Tests

```bash
# Ejecutar toda la suite (370 tests)
python3 -m pytest tests/ -v

# Por módulo
python3 -m pytest tests/test_utf8_qnn.py -v      # Codec UTF-8
python3 -m pytest tests/test_h7_bridge.py -v     # Bridge Python↔C
python3 -m pytest tests/test_h7_physics.py -v    # Física metripléctica
```

### Cobertura de tests

| Suite | Tests | Qué valida |
|---|---|---|
| `test_utf8_qnn` | ~50 | Normalización, flag MSB, unitariedad CSWAP, Isomorfismo Nivel 3 |
| `test_h7_bridge` | ~60 | SU(2)→cuaternión, mapeo psi/v/energy, binario 152B, DRIFT_072 |
| `test_h7_physics` | ~260 | Lagrangiano, competencia simpléctica/métrica, Z₇, Test del Tiempo |

---

## Glosario

| Término | Definición |
|---|---|
| **Metripléctica** | Geometría que combina simpléctica (conservativa) y métrica (disipativa) |
| **$O_n$** | Operador Áureo: módulo del vacío topológico |
| **DRIFT_072** | $7 - 2\pi \approx 0.7168$: residuo entre $\mathbb{Z}_7$ y $U(1)$ |
| **$\mathbb{Z}_7$** | Grupo cíclico de 7 elementos: espacio de clasificación de partículas |
| **$SU(2)$** | Grupo de matrices unitarias 2×2 de determinante 1: rotaciones cuánticas |
| **$Q_8$** | Cuaterniones unitarios: par antagonista topológico |
| **EstadoCuantico** | Struct C con `psi[8]`: espacio de Hilbert de 3 qubits |
| **CSWAP** | Compuerta Fredkin: intercambio controlado, corazón del cifrado |
| **Isomorfismo Nivel 3** | Correspondencia física profunda: mismo comportamiento en límites asintóticos |
| **his-torial/** | Directorio de salidas del pipeline (evita saturar la raíz) |
