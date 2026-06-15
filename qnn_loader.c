/**
 * h7_loader.c
 * Consumidor C del binary export de h7_bridge.py
 * Carga h7_state_nN.bin → MetriplecticState + TorsionObservables + EstadoCuantico
 * Compilar: gcc -o h7_loader h7_loader.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "metriplectic.h"

/* Layout binario: 152 bytes little-endian
   [0..6]  : MetriplecticState  (psi, v, energy, q.w, q.x, q.y, q.z)
   [7..10] : TorsionObservables (energy_density, entropy_gradient, spatial_torsion, chirality)
   [11..18]: EstadoCuantico.psi[8]
*/
#define BINARY_DOUBLES 19

typedef struct {
    MetriplecticState state;
    TorsionObservables torsion;
    EstadoCuantico quantum;
} H7FullExport;

int h7_load_binary(const char *path, H7FullExport *out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[h7_loader] No se pudo abrir: %s\n", path); return -1; }

    double buf[BINARY_DOUBLES];
    size_t read = fread(buf, sizeof(double), BINARY_DOUBLES, f);
    fclose(f);

    if (read != BINARY_DOUBLES) {
        fprintf(stderr, "[h7_loader] Archivo incompleto: leídos %zu de %d doubles\n",
                read, BINARY_DOUBLES);
        return -2;
    }

    /* MetriplecticState */
    out->state.psi    = buf[0];
    out->state.v      = buf[1];
    out->state.energy = buf[2];
    out->state.q.w    = buf[3];
    out->state.q.x    = buf[4];
    out->state.q.y    = buf[5];
    out->state.q.z    = buf[6];

    /* TorsionObservables */
    out->torsion.energy_density   = buf[7];
    out->torsion.entropy_gradient = buf[8];
    out->torsion.spatial_torsion  = buf[9];   /* DRIFT_072 = 7-2π */
    out->torsion.chirality        = buf[10];

    /* EstadoCuantico */
    for (int i = 0; i < HILBERT_DIM; i++)
        out->quantum.psi[i] = buf[11 + i];

    return 0;
}

void h7_print_export(const H7FullExport *e) {
    printf("\n=== H7 Export (Python → C) ===\n");
    printf("MetriplecticState:\n");
    printf("  psi    = %.8f  (cos(πφn) quasiperiod)\n", e->state.psi);
    printf("  v      = %.8f  (cos(πn)  parity)\n",      e->state.v);
    printf("  energy = %.8f  (Ψn classifier)\n",         e->state.energy);
    printf("  q      = (w=%.6f, x=%.6f, y=%.6f, z=%.6f)\n",
           e->state.q.w, e->state.q.x, e->state.q.y, e->state.q.z);

    printf("\nTorsionObservables:\n");
    printf("  energy_density   = %.8f\n", e->torsion.energy_density);
    printf("  entropy_gradient = %.8f  (Shannon normalizado)\n", e->torsion.entropy_gradient);
    printf("  spatial_torsion  = %.10f  (DRIFT_072 = 7-2π)\n", e->torsion.spatial_torsion);
    printf("  chirality        = %.8f  (paridad Z7)\n", e->torsion.chirality);

    printf("\nEstadoCuantico psi[8]:\n  ");
    for (int i = 0; i < HILBERT_DIM; i++)
        printf("%.4f ", e->quantum.psi[i]);
    printf("\n");

    /* golden_operator sobre psi como verificación cruzada */
    printf("\ngolden_operator check:\n");
    printf("  golden_operator(1) = %.8f\n", golden_operator(1));
    printf("  psi × v            = %.8f  (debe aproximar energy - v)\n",
           e->state.psi * e->state.v);
}

/* forward_pass stub: demonstra el flujo QNN con datos H7 */
void demo_forward_pass(const H7FullExport *e) {
    QNNGrid grid = initialize_qnn_grid();
    EstadoCuantico output;

    double On = golden_operator(1);  /* n=1 demo; en prod. pasar n real */
    forward_pass(&grid, (EstadoCuantico *)&e->quantum, &output, On);

    printf("\nQNN forward_pass output psi[8]:\n  ");
    for (int i = 0; i < HILBERT_DIM; i++)
        printf("%.4f ", output.psi[i]);
    printf("\n");
}

/* Codec UTF-8 a Cuántico */
char decode_char(EstadoCuantico estado) {
    double temp[8];
    for(int i=0; i<8; i++) temp[i] = estado.psi[i];
    
    // 1. Inversa de CX(1, 0)
    // Qubit 1 es control, Qubit 0 es target
    // Intercambia estados 2<->3 y 6<->7
    double swap;
    swap = temp[2]; temp[2] = temp[3]; temp[3] = swap;
    swap = temp[6]; temp[6] = temp[7]; temp[7] = swap;
    
    // 2. Inversa de CSWAP(0, 1, 2)
    // Qubit 0 es control, Qubit 1 y 2 targets
    // Intercambia estados 3<->5
    swap = temp[3]; temp[3] = temp[5]; temp[5] = swap;
    
    // 3. Inversa de H(0)
    // Mezcla pares (0,1), (2,3), (4,5), (6,7)
    for(int i=0; i<8; i+=2) {
        double a = temp[i];
        double b = temp[i+1];
        temp[i]   = (a + b) / sqrt(2.0);
        temp[i+1] = (a - b) / sqrt(2.0);
    }
    
    char out_char = 0;
    for(int i=0; i<8; i++) {
        if(temp[i] > 0.1) {
            // Reconstruimos el bit (ignorando el bit 0 que es el Flag de Seguridad)
            if(i != 0) {
                out_char |= (1 << (7 - i));
            }
        }
    }
    return out_char;
}

EstadoCuantico encode_char(char c) {
    // Stub for C-side encoding (feed-forward)
    EstadoCuantico ec = {0};
    // TODO: implement C-side initialization if needed
    return ec;
}

QNNGrid initialize_qnn_grid() {
    QNNGrid grid = {0};
    return grid;
}

void forward_pass(QNNGrid *grid, EstadoCuantico *input, EstadoCuantico *output, double On) {
    // Stub
    for(int i=0; i<8; i++) {
        output->psi[i] = input->psi[i] * On;
    }
}

int main(int argc, char *argv[]) {
    const char *path = (argc > 1) ? argv[1] : "his-torial/h7_state_char_A.bin";

    H7FullExport export_data;
    if (h7_load_binary(path, &export_data) != 0)
        return 1;

    h7_print_export(&export_data);
    demo_forward_pass(&export_data);
    
    // Intentar descifrar como carácter UTF-8
    char recuperado = decode_char(export_data.quantum);
    printf("\n[Decodificador UTF-8 Cuántico]\n");
    printf("  Carácter recuperado del Statevector: '%c' (ASCII: %d)\n", recuperado, (int)recuperado);

    return 0;
}