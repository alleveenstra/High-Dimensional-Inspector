#ifndef HDI_WRAPPER_H
#define HDI_WRAPPER_H

#include <stdexcept>

class PyHDIException : public std::runtime_error {
public:
    explicit PyHDIException(const std::string &__arg) : runtime_error(__arg){}
};

class HDILogger {
public:
    virtual void log(int level, std::string msg) = 0;
    HDILogger() = default;
    virtual ~HDILogger() = default;
    enum {
        INFO,
        WARNING,
        ERROR
    };
};

class HDIParameters {
protected:
    HDILogger *_logger = nullptr;

    float *_source = nullptr;
    float *_target = nullptr;
    unsigned int _n_points = 0;
    unsigned int _source_dimensions = 0;
    unsigned int _target_dimensions = 0;

    float _perplexity = 30.0;
    int _seed = 0;
    float _minimum_gain = 0.1;
    float _eta = 200.;
    float _momentum = 0.5;
    float _final_momentum = 0.8;
    float _mom_switching_iter = 250;
    float _exaggeration_factor = 4.0;
    unsigned int _remove_exaggeration_iter = 250;
    float _theta = 0.5;

public:

    HDIParameters() = default;
    ~HDIParameters() = default;

    void set_logger(HDILogger *logger);

    void del_logger();

    void set_n_points(unsigned int value);

    void set_input(float *matrix, unsigned int n_points, unsigned int dimensions);

    void set_output(float *matrix, unsigned int n_points, unsigned int dimensions);

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
    HDIParameters _parameters;
public:
    HDIParameters &parameters();

    HDItSNE() = default;
    ~HDItSNE() = default;

    void run(unsigned int iterations);
};

class HDIaSNE {
    HDIParameters _parameters;
public:
    HDIParameters &parameters();

    HDIaSNE() = default;
    ~HDIaSNE() = default;

    void run(unsigned int iterations);
};

#endif
