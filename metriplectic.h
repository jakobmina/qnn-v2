#ifndef METRIPLECTIC_H
#define METRIPLECTIC_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 *  METRIPLÉX QNN- Unified Autonomous Edition
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PHI 1.618033988749895
#define HILBERT_DIM 8
#define QNN_LAYERS 4

/* --- Estructuras Algebraicas --- */
typedef struct { double w, x, y, z; } Cuaternion;

/* Simulación de Oscilador (circuit.c) */
typedef struct {
    double psi;
    double v;
    double energy;
    Cuaternion q;
} MetriplecticState;

/* Actor Generativo (AI Cursor) */
typedef struct {
    double x, y;   /* Posición en el plano de la pantalla */
    double vx, vy; /* Velocidad (Momento) */
    double energy; /* Estabilidad del trazo */
} GenerativeActor;

typedef struct {
    double H;
    double S;
} LagrangianComponents;

/* --- Regla 2.1: Operador Áureo --- */
static inline double golden_operator(double n) {
    return cos(M_PI * n) * cos(M_PI * PHI * n);
}

/* --- Estructuras QNN y Sensing --- */
typedef struct {
    double energy_density;
    double entropy_gradient;
    double spatial_torsion;
    double chirality;
} TorsionObservables;

typedef struct {
    double weight;
    double bias;
    int pair_indices[2];
} QNNLayer;

typedef struct {
    QNNLayer layers[QNN_LAYERS];
    double learning_rate;
} QNNGrid;

typedef struct {
    double psi[HILBERT_DIM];
} EstadoCuantico;

/* --- Prototipos de Actor Generativo --- */
GenerativeActor update_actor(GenerativeActor actor, TorsionObservables feedback, double On);

/* --- Prototipos de Sistema --- */
LagrangianComponents compute_lagrangian(MetriplecticState *u);
TorsionObservables feel_screen(uint8_t *buffer, int width, int height);
EstadoCuantico map_observables_to_hilbert(TorsionObservables obs);
QNNGrid initialize_qnn_grid();
void forward_pass(QNNGrid *grid, EstadoCuantico *input, EstadoCuantico *output, double On);

/* Codec */
EstadoCuantico encode_char(char c);
char decode_char(EstadoCuantico estado);

#endif /* METRIPLECTIC_H */