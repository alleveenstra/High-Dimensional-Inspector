%module(directors="1") pyhdi

%feature("director") HDILogger;

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

%include "std_string.i"
%include "numpy.i"
%include "exception.i"

class HDILogger {
public:
    virtual void log(int level, std::string msg) = 0;
    HDILogger();
    virtual ~HDILogger();
};

class HDIParameters {
public:
    HDIParameters();
    ~HDIParameters();

    void set_n_points(unsigned int value);

    %apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float *matrix, unsigned int n_points, unsigned int dimensions)};
    void set_input(float *matrix, unsigned int n_points, unsigned int dimensions);

    %apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float *matrix, unsigned int n_points, unsigned int dimensions)};
    void set_output(float *matrix, unsigned int n_points, unsigned int dimensions);

    void set_logger(HDILogger *logger);

    void del_logger();

    void set_perplexity(float value);

    void set_seed(int value);

    void set_minimum_gain(float value);

    void set_eta(float value);

    void set_theta(float value);

    void set_momentum(float value);

    void set_final_momentum(float value);

    void set_mom_switching_iter(float value);

    void set_exaggeration_factor(float value);

    void set_remove_exaggeration_iter(unsigned int value);

    friend class HDItSNE;
    friend class HDIaSNE;
};

class HDItSNE {
public:
    HDItSNE();
    ~HDItSNE();

    HDIParameters &parameters();

    void run(unsigned int iterations);

};

class HDIaSNE {
public:
    HDIaSNE();
    ~HDIaSNE();

    HDIParameters &parameters();

    void run(unsigned int iterations);

};

%pythoncode %{
PyHDIException = _pyhdi.PyHDIException
%}
