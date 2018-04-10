#ifndef HDI_WRAPPER_H
#define HDI_WRAPPER_H

#include <stdexcept>

class PyHDIException : public std::runtime_error {
public:
    explicit PyHDIException(const std::string &__arg) : runtime_error(__arg){}
};

class HDI_Logger {
public:
    virtual void log(int level, std::string msg) = 0;
    virtual ~HDI_Logger() {};
    enum {
        INFO,
        WARNING,
        ERROR
    };
};

class HDI_Parameters {
protected:
    HDI_Logger *_logger = nullptr;

    double *_source = nullptr;
    double *_target = nullptr;
    unsigned int _n_points = 0;
    unsigned int _source_dimensions = 0;
    unsigned int _target_dimensions = 0;

    double _perplexity = 30.0;
    int _seed = 0;
    double _minimum_gain = 0.1;
    double _eta = 200.;
    double _momentum = 0.5;
    double _final_momentum = 0.8;
    double _mom_switching_iter = 250;
    double _exaggeration_factor = 4.0;
    unsigned int _remove_exaggeration_iter = 250;
    double _theta = 0.5;

public:
    void set_logger(HDI_Logger *logger);

    void del_logger();

    void set_n_points(unsigned int value);

    void set_input(double *matrix, unsigned int n_points, unsigned int dimensions);

    void set_output(double *matrix, unsigned int n_points, unsigned int dimensions);

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
    HDI_Parameters _parameters;
public:
    HDI_Parameters &parameters();

    void run(unsigned int iterations);
};

class HDI_aSNE {
    HDI_Parameters _parameters;
public:
    HDI_Parameters &parameters();

    void run(unsigned int iterations);
};

#endif
