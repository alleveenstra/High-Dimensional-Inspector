%module pyhdi

%init %{
import_array();

pPyHDIException = PyErr_NewException("_pyhdi.PyHDIException", NULL, NULL);
Py_INCREF(pPyHDIException);
PyModule_AddObject(m, "PyHDIException", pPyHDIException);
%}

%exception {
    try {
        $action
    } catch (PyHDIException &e) {
        PyErr_SetString(pPyHDIException, const_cast<char*>(e.what()));
        SWIG_fail;
    }
}

%{
#define SWIG_FILE_WITH_INIT
#include "libpyhdi.h"
static PyObject* pPyHDIException;
%}

%include "numpy.i"
%include "exception.i"

void set_n_points(unsigned int value);

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *xx, unsigned int n_points, unsigned int dimensions)};
void set_input(double *xx, unsigned int n_points, unsigned int dimensions);

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *yy, unsigned int n_points, unsigned int dimensions)};
void set_output(double *yy, unsigned int n_points, unsigned int dimensions);

void run_tsne(double perplexity,
              int seed,
              double minimum_gain,
              double eta,
              double momentum,
              double final_momentum,
              double mom_switching_iter,
              double exaggeration_factor,
              unsigned int remove_exaggeration_iter,
              unsigned int iterations);

void run_asne(double perplexity,
              double theta,
              unsigned int exaggeration_iter,
              unsigned int iterations);

%pythoncode %{
    PyHDIException = _pyhdi.PyHDIException
%}
