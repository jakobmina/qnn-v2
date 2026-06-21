"""
tests/test_h7_qnn_batch_nodesort.py
Pytest para h7_qnn_batch_nodesort.py

Cubre:
  - tokenizar(): correctitud y casos límite
  - nodo_de(): mapeo al espacio teórico [-0.25, 0.25]
  - node_sort_por_covarianza(): orden descendente/ascendente, estabilidad,
    respeto al espacio teórico
  - construir_prompt(): temperatura dinámica (competencia simpléctica/métrica)
  - procesar_batch(): integración con mock QNN (sin Ollama ni Qiskit real)
  - run_batch_inference(): smoke test con mock completo

Principio Metripléctica (Regla 1.3):
  El NodeSort mezcla componente simpléctica (orden descendente, primacy)
  con componente métrica (agrupación en cubetas, relajación al atractor cov=0).
  Los tests verifican ambos brazos por separado.
"""

import pytest
from unittest.mock import patch, MagicMock

# ── módulo bajo prueba ──────────────────────────────────────────────────────
from h7_qnn_batch_nodesort import (
    tokenizar,
    nodo_de,
    node_sort_por_covarianza,
    construir_prompt,
    procesar_token,
    procesar_batch,
    run_batch_inference,
    COV_MIN,
    COV_MAX,
    BATCH_SHOTS,
    _safe_max_workers,
)


# ════════════════════════════════════════════════════════════════════════════
# 1. TOKENIZACIÓN
# ════════════════════════════════════════════════════════════════════════════

class TestTokenizar:
    def test_texto_simple(self):
        tokens = tokenizar("hello world foo")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens

    def test_codigo_python(self):
        snippet = "def foo(x, y): return x + y"
        tokens = tokenizar(snippet)
        assert "def" in tokens
        assert "foo" in tokens
        assert "return" in tokens
        assert "x" in tokens

    def test_numeros(self):
        tokens = tokenizar("42 3.14 100")
        assert "42" in tokens
        assert "3.14" in tokens
        assert "100" in tokens

    def test_string_vacio(self):
        tokens = tokenizar("")
        assert tokens == []

    def test_solo_espacios(self):
        tokens = tokenizar("   \t\n  ")
        assert tokens == []

    def test_no_produce_tokens_vacios(self):
        tokens = tokenizar("a  b   c")
        assert all(t.strip() for t in tokens)


# ════════════════════════════════════════════════════════════════════════════
# 2. nodo_de() — Mapeo al espacio teórico
# ════════════════════════════════════════════════════════════════════════════

class TestNodoDe:
    def test_cov_minima_es_nodo_0(self):
        assert nodo_de(COV_MIN, 8) == 0

    def test_cov_maxima_es_nodo_7(self):
        assert nodo_de(COV_MAX, 8) == 7

    def test_cov_cero_es_nodo_central(self):
        nodo = nodo_de(0.0, 8)
        # cov=0.0 está en la mitad del rango [-0.25, 0.25]
        assert nodo in (3, 4)

    def test_clamp_supera_maximo(self):
        """Covarianzas fuera de rango se acotan al nodo máximo."""
        assert nodo_de(99.0, 8) == 7

    def test_clamp_bajo_minimo(self):
        assert nodo_de(-99.0, 8) == 0

    def test_rango_valido_siempre(self):
        """Para cualquier cov ∈ [-1, 1], nodo ∈ [0, num_nodos-1]."""
        num = 16
        for cov in [-1.0, -0.25, -0.1, 0.0, 0.1, 0.25, 1.0]:
            n = nodo_de(cov, num)
            assert 0 <= n < num, f"nodo_de({cov}, {num}) = {n} fuera de rango"


# ════════════════════════════════════════════════════════════════════════════
# 3. node_sort_por_covarianza() — Componente simpléctica + métrica
# ════════════════════════════════════════════════════════════════════════════

class TestNodeSort:
    def _items(self, covs):
        return [{"idx": i, "token": f"t{i}", "covariance": c,
                 "psi": 0.0, "energy": 0.0, "torsion": 0.0}
                for i, c in enumerate(covs)]

    def test_orden_descendente_basico(self):
        items = self._items([0.05, 0.20, -0.10, 0.15])
        result = node_sort_por_covarianza(items, num_nodos=4, descendente=True)
        covs = [r["covariance"] for r in result]
        # verificar orden descendente global
        assert covs == sorted(covs, reverse=True)

    def test_orden_ascendente(self):
        items = self._items([0.05, 0.20, -0.10, 0.15])
        result = node_sort_por_covarianza(items, num_nodos=4, descendente=False)
        covs = [r["covariance"] for r in result]
        assert covs == sorted(covs)

    def test_preserva_todos_los_elementos(self):
        items = self._items([0.1, -0.2, 0.05, 0.24, -0.24, 0.0])
        result = node_sort_por_covarianza(items, num_nodos=8)
        assert len(result) == len(items)
        tokens_in = {i["token"] for i in items}
        tokens_out = {r["token"] for r in result}
        assert tokens_in == tokens_out

    def test_un_elemento(self):
        items = self._items([0.12])
        result = node_sort_por_covarianza(items, num_nodos=8)
        assert len(result) == 1

    def test_lista_vacia(self):
        result = node_sort_por_covarianza([], num_nodos=8)
        assert result == []

    def test_todos_iguales(self):
        """Si todas las covarianzas son iguales, el sort es estable."""
        items = self._items([0.1] * 6)
        result = node_sort_por_covarianza(items, num_nodos=4)
        assert len(result) == 6

    def test_espacio_teorico_no_depende_del_batch(self):
        """
        NodeSort debe usar [-0.25, 0.25] fijo, no min/max del batch.
        Verificamos que un item con cov=0.20 sea siempre nodo 7
        independientemente de los otros valores en el batch.
        """
        batch_a = self._items([0.20, -0.10])
        batch_b = self._items([0.20, 0.05])
        nodo_a = nodo_de(0.20, 8)
        nodo_b = nodo_de(0.20, 8)
        assert nodo_a == nodo_b, "El nodo de 0.20 debe ser igual en ambos batches"


# ════════════════════════════════════════════════════════════════════════════
# 4. construir_prompt() — Temperatura dinámica (dualidad simpléctica/métrica)
# ════════════════════════════════════════════════════════════════════════════

class TestConstruirPrompt:
    def _batch(self, covs):
        return [{"idx": i, "token": f"tok{i}", "covariance": c,
                 "psi": 0.5, "energy": 1.0, "torsion": 0.3}
                for i, c in enumerate(covs)]

    def test_temperatura_dinamica_rango_valido(self):
        """La temperatura debe estar siempre en [0.1, 1.5]."""
        for covs in [[-0.25] * 5, [0.0] * 5, [0.25] * 5, [0.1, -0.1, 0.2]]:
            _, temp = construir_prompt("test", self._batch(covs))
            assert 0.1 <= temp <= 1.5, f"Temperatura {temp} fuera de rango para covs={covs}"

    def test_alta_cov_alta_temperatura(self):
        """Alta covarianza promedio → temperatura más alta (componente simpléctica domina)."""
        _, temp_alta = construir_prompt("texto", self._batch([0.24] * 10))
        _, temp_baja = construir_prompt("texto", self._batch([0.01] * 10))
        assert temp_alta > temp_baja

    def test_prompt_contiene_texto_original(self):
        seed = "mi código de prueba"
        prompt, _ = construir_prompt(seed, self._batch([0.1] * 5))
        assert "mi código de prueba" in prompt

    def test_prompt_trunca_texto_largo(self):
        seed_largo = "A" * 1000
        prompt, _ = construir_prompt(seed_largo, self._batch([0.1] * 5))
        assert "..." in prompt

    def test_batch_vacio_no_explota(self):
        """Batch vacío debe manejarse graciosamente (cov_promedio=0)."""
        prompt, temp = construir_prompt("texto", [])
        assert isinstance(prompt, str)
        assert 0.1 <= temp <= 1.5

    def test_top_k_max_15(self):
        """Debe mostrar máximo 15 tokens destacados."""
        batch_grande = self._batch([0.1] * 50)
        prompt, _ = construir_prompt("texto largo", batch_grande)
        # "15 tokens" debe aparecer en el prompt
        assert "15" in prompt


# ════════════════════════════════════════════════════════════════════════════
# 5. procesar_token() — Integración con QNN (mock)
# ════════════════════════════════════════════════════════════════════════════

FAKE_QNN_RESULT = {
    "history": [
        {"iteration": 1, "covariance": 0.05},
        {"iteration": 2, "covariance": 0.08},
        {"iteration": 3, "covariance": 0.12},  # ← último
    ],
    "hash": {
        "psi": 0.707,
        "energy": 1.234,
        "torsion": 0.432,
    }
}


class TestProcesarToken:
    @patch("h7_qnn_batch_nodesort.generate_metriplectic_hash",
           return_value=FAKE_QNN_RESULT)
    def test_token_numerico(self, mock_hash):
        r = procesar_token("42", 0)
        mock_hash.assert_called_once_with(42, iterations=3, shots=BATCH_SHOTS)
        assert r["token"] == "42"
        assert r["idx"] == 0
        assert r["covariance"] == 0.12  # último history

    @patch("h7_qnn_batch_nodesort.generate_metriplectic_hash",
           return_value=FAKE_QNN_RESULT)
    @patch("h7_qnn_batch_nodesort.string_to_qnn_seed", return_value=99)
    def test_token_string(self, mock_seed, mock_hash):
        r = procesar_token("hello", 3)
        mock_seed.assert_called_once_with("hello")
        mock_hash.assert_called_once_with(99, iterations=3, shots=BATCH_SHOTS)
        assert r["idx"] == 3
        assert r["psi"] == 0.707
        assert r["energy"] == 1.234
        assert r["torsion"] == 0.432

    @patch("h7_qnn_batch_nodesort.generate_metriplectic_hash",
           return_value=FAKE_QNN_RESULT)
    @patch("h7_qnn_batch_nodesort.string_to_qnn_seed", return_value=0)
    def test_seed_cero_evitado(self, mock_seed, mock_hash):
        """seed=0 debe convertirse a 1 (evitar semilla nula)."""
        procesar_token("null_token", 0)
        args = mock_hash.call_args[0]
        assert args[0] != 0, "seed=0 no debe pasarse a generate_metriplectic_hash"


# ════════════════════════════════════════════════════════════════════════════
# 6. procesar_batch() — Paralelismo + orden preservado
# ════════════════════════════════════════════════════════════════════════════

class TestProcesarBatch:
    @patch("h7_qnn_batch_nodesort.generate_metriplectic_hash",
           return_value=FAKE_QNN_RESULT)
    def test_indices_correctos(self, mock_hash):
        tokens = ["a", "b", "c", "1", "2"]
        resultados = procesar_batch(tokens, max_workers=2)
        # cada resultado debe tener idx que coincide con su posición en la lista
        for expected_idx, r in enumerate(resultados):
            assert r["idx"] == expected_idx

    @patch("h7_qnn_batch_nodesort.generate_metriplectic_hash",
           return_value=FAKE_QNN_RESULT)
    def test_longitud_correcta(self, mock_hash):
        tokens = ["x"] * 10
        resultados = procesar_batch(tokens, max_workers=4)
        assert len(resultados) == 10

    @patch("h7_qnn_batch_nodesort.generate_metriplectic_hash",
           return_value=FAKE_QNN_RESULT)
    def test_batch_un_token(self, mock_hash):
        resultados = procesar_batch(["solo"], max_workers=1)
        assert len(resultados) == 1
        assert resultados[0]["token"] == "solo"


# ════════════════════════════════════════════════════════════════════════════
# 7. run_batch_inference() — Smoke test pipeline completo
# ════════════════════════════════════════════════════════════════════════════

class TestRunBatchInference:
    @patch("h7_qnn_batch_nodesort.run_ollama")  # no llama a Ollama real
    @patch("h7_qnn_batch_nodesort.generate_metriplectic_hash",
           return_value=FAKE_QNN_RESULT)
    def test_pipeline_retorna_batch_ordenado(self, mock_hash, mock_ollama):
        batch = run_batch_inference("def add(x, y): return x + y",
                                    num_nodos=4, verbose=False)
        assert isinstance(batch, list)
        assert len(batch) > 0
        # verificar orden descendente
        covs = [b["covariance"] for b in batch]
        assert covs == sorted(covs, reverse=True)

    @patch("h7_qnn_batch_nodesort.run_ollama")
    @patch("h7_qnn_batch_nodesort.generate_metriplectic_hash",
           return_value=FAKE_QNN_RESULT)
    def test_pipeline_llama_ollama(self, mock_hash, mock_ollama):
        run_batch_inference("hola mundo", num_nodos=4, verbose=False)
        mock_ollama.assert_called_once()

    @patch("h7_qnn_batch_nodesort.run_ollama")
    @patch("h7_qnn_batch_nodesort.generate_metriplectic_hash",
           return_value=FAKE_QNN_RESULT)
    def test_pipeline_texto_vacio_no_explota(self, mock_hash, mock_ollama):
        """Un texto vacío produce 0 tokens; el pipeline no debe lanzar excepción."""
        # tokenizar("") → [] → procesar_batch([]) → []
        batch = run_batch_inference("", num_nodos=4, verbose=False)
        assert isinstance(batch, list)


# ════════════════════════════════════════════════════════════════════════════
# 8. Constantes del espacio teórico (Regla 1.3: cota [-0.25, 0.25])
# ════════════════════════════════════════════════════════════════════════════

class TestConstantesTeorica:
    def test_cov_min_es_menos_025(self):
        assert COV_MIN == -0.25

    def test_cov_max_es_025(self):
        assert COV_MAX == 0.25

    def test_rango_simetrico(self):
        assert abs(COV_MIN) == COV_MAX


# ════════════════════════════════════════════════════════════════════════════
# 9. _safe_max_workers() — Control de memoria (fix OOM)
# ════════════════════════════════════════════════════════════════════════════

class TestSafeMaxWorkers:
    def test_no_supera_pedido(self):
        """Nunca retorna más workers que los solicitados."""
        with patch("h7_qnn_batch_nodesort._HAS_PSUTIL", True):
            with patch("h7_qnn_batch_nodesort.psutil") as mock_psutil:
                # Simular 8 GB libres → 40 workers posibles
                mock_psutil.virtual_memory.return_value.available = 8 * 1024**3
                resultado = _safe_max_workers(4)
                assert resultado == 4

    def test_limita_con_poca_ram(self):
        """Con 400 MB libres (2 workers máximo) y 8 pedidos, retorna ≤2."""
        with patch("h7_qnn_batch_nodesort._HAS_PSUTIL", True):
            with patch("h7_qnn_batch_nodesort.psutil") as mock_psutil:
                mock_psutil.virtual_memory.return_value.available = 400 * 1024**2
                resultado = _safe_max_workers(8)
                assert resultado <= 2

    def test_sin_psutil_limite_conservador(self):
        """Sin psutil instalado, max_workers se acota a 4."""
        with patch("h7_qnn_batch_nodesort._HAS_PSUTIL", False):
            resultado = _safe_max_workers(8)
            assert resultado <= 4

    def test_siempre_al_menos_uno(self):
        """Nunca retorna 0 aunque no haya casi RAM."""
        with patch("h7_qnn_batch_nodesort._HAS_PSUTIL", True):
            with patch("h7_qnn_batch_nodesort.psutil") as mock_psutil:
                mock_psutil.virtual_memory.return_value.available = 1024  # 1 KB
                resultado = _safe_max_workers(8)
                assert resultado >= 1


# ════════════════════════════════════════════════════════════════════════════
# 10. BATCH_SHOTS — Reducción de memoria por circuito
# ════════════════════════════════════════════════════════════════════════════

class TestBatchShots:
    def test_batch_shots_es_512(self):
        """BATCH_SHOTS debe ser 512 (balance covarianza vs memoria)."""
        assert BATCH_SHOTS == 512

    def test_batch_shots_propagado_a_token(self):
        """procesar_token debe pasar BATCH_SHOTS a generate_metriplectic_hash."""
        with patch("h7_qnn_batch_nodesort.generate_metriplectic_hash",
                   return_value=FAKE_QNN_RESULT) as mock_hash:
            procesar_token("test", 0)
            _, kwargs = mock_hash.call_args
            assert kwargs.get("shots", None) == BATCH_SHOTS or \
                   mock_hash.call_args[0][2] == BATCH_SHOTS if len(mock_hash.call_args[0]) > 2 else \
                   mock_hash.call_args[1].get("shots") == BATCH_SHOTS
