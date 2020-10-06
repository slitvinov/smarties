#ifdef __cplusplus
extern "C" {
#endif

void smarties_sendinitstate_ (uintptr_t, double *S, int *dim, int *);
void smarties_sendtermstate_ (uintptr_t, double *S, int *dim, double *R, int *);
void smarties_sendstate_ (uintptr_t, double *S, int *dim, double *R, int *);
void smarties_recvaction_ (uintptr_t, double *A, int *dim, int *);
void smarties_setactionscales_ (uintptr_t, double *hi, double *lo, int *are_bounds, int *dim, int *);
void smarties_setstateobservable_ (uintptr_t, int *bobs, int *dim, int *);
void smarties_setstatescales_ (uintptr_t, double *hi, double *lo, int *dim, int *);
void smarties_setstateactiondims_ (uintptr_t *, int *sdim, int *adim, int *);

#ifdef __cplusplus
}
#endif
