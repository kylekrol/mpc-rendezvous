#include "ecos.h"
#include "mpc_controller.h"

#include <stdlib.h>

#define MPC_N
#define MPC_M
#define MPC_P
#define MPC_L
#define MPC_NCONES

struct mpc {
    idxint *q;
    pfloat *Gpr;
    idxint *Gjc, Gir;
    pfloat *Apr;
    idxint *Ajc, Air;
    pfloat *c, *h, *b;
};

mpc_t mpc_init(int T, double umax, double R, double Q, double const *A, double const *B) {
    idxint n, m, p, l;
    idxint ncones;

    // TODO
}

void mpc_free(mpc_t self) {
    free(self->q);
    free(self->Gpr);
    free(self->Gjc);
    free(self->Gir);
    free(self->Apr);
    free(self->Ajc);
    free(self->Air);
    free(self->c);
    free(self->h);
    free(self->b);
    free(self);
}

int mpc_solve(mpc_t self, mpc_solve_in_t in, mpc_solve_out_t out);
