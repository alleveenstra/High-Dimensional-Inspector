%module(directors="1") pyhdi

%feature("director") HDI_Logger;

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
#include "hdi_wrapper.h"
static PyObject* pPyHDIException;
%}

%include "numpy.i"
%include "exception.i"

class HDI_Logger {
public:
    virtual void log() = 0;
    virtual ~HDI_Logger() {};
};

class HDI_Parameters {
public:
    void set_n_points(unsigned int value);

    %apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *matrix, unsigned int n_points, unsigned int dimensions)};
    void set_input(double *matrix, unsigned int n_points, unsigned int dimensions);

    %apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *matrix, unsigned int n_points, unsigned int dimensions)};
    void set_output(double *matrix, unsigned int n_points, unsigned int dimensions);

    void set_logger(HDI_Logger *logger);

    void set_perplexity(double value);

    void set_seed(int value);

    void set_minimum_gain(double value);

    void set_eta(double value);

    void set_theta(double value);

    void set_momentum(double value);

    void set_final_momentum(double value);

    void set_mom_switching_iter(double value);

    void set_exaggeration_factor(double value);

    void set_remove_exaggeration_iter(unsigned int value);

    friend class HDI_tSNE;
    friend class HDI_aSNE;
};

class HDI_tSNE {
public:
    HDI_Parameters &parameters();

    void run(unsigned int iterations);

};

class HDI_aSNE {
public:
    HDI_Parameters &parameters();

    void run(unsigned int iterations);

};

%pythoncode %{
PyHDIException = _pyhdi.PyHDIException
%}
