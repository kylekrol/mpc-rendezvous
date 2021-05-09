#ifndef MPC_H_
#define MPC_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mpc *mpc_t;

mpc_t mpc_init(int T, double umax, double R, double Q, double const *A, double const *B);
void mpc_free(mpc_t self);

typedef struct {
  double const *x;
} mpc_solve_in_t;

typedef struct {
  double *L, *u, *X, *U;
} mpc_solve_out_t;

int mpc_solve(mpc_t self, mpc_solve_in_t in, mpc_solve_out_t out);

#ifdef __cplusplus
}
#endif

#endif
