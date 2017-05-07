#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

extern SEXP n1PDF(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                  SEXP, SEXP);
extern SEXP n1PDF_plba1(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                        SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP n1PDF_plba2(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                        SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP n1PDF_plba3(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                        SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

extern SEXP histc_entry(SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"n1PDF",       (DL_FUNC) &n1PDF, 12},
  {"n1PDF_plba1", (DL_FUNC) &n1PDF_plba1, 15},
  {"n1PDF_plba2", (DL_FUNC) &n1PDF_plba2, 16},
  {"n1PDF_plba3", (DL_FUNC) &n1PDF_plba3, 18},
  {"histc_entry", (DL_FUNC) &histc_entry, 5},
  {NULL, NULL, 0}
};

void R_init_CircularDDM(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}

