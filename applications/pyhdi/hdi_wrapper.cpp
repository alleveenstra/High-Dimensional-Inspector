#include <iostream>

#include "hdi/dimensionality_reduction/tsne.h"
#include "hdi/utils/cout_log.h"
#include "hdi/data/empty_data.h"
#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/dimensionality_reduction/sparse_tsne_user_def_probabilities.h"
#include "hdi/utils/visual_utils.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/analytics/multiscale_embedder_system_qobj.h"

#include "hdi_wrapper.h"

void HDIParameters::set_logger(HDILogger *logger) {
    _logger = logger;
}

void HDIParameters::del_logger() {
    delete _logger;
    _logger = nullptr;
}

void HDIParameters::set_n_points(unsigned int value) {
    _n_points = value;
}

void HDIParameters::set_input(double *matrix, unsigned int n_input_points, unsigned int dimensions) {
    if (_n_points == 0) {
        throw PyHDIException("call set_n_points() before set_input()");
    }
    if (n_input_points != _n_points) {
        std::stringstream fmt;
        fmt << "n_points not equal to matrix n_input_points, " << _n_points << " != " << n_input_points;
        throw PyHDIException(fmt.str());
    }
    _source = matrix;
    _source_dimensions = dimensions;
}

void HDIParameters::set_output(double *matrix, unsigned int n_output_points, unsigned int dimensions) {
    if (_n_points == 0) {
        throw PyHDIException("call set_n_points() before set_output()");
    }
    if (n_output_points != _n_points) {
        std::stringstream fmt;
        fmt << "n_points not equal to matrix n_output_points, " << _n_points << " != " << n_output_points;
        throw PyHDIException(fmt.str());
    }
    _target = matrix;
    _target_dimensions = dimensions;
}

void HDIParameters::set_perplexity(double value) {
    _perplexity = value;
}

void HDIParameters::set_seed(int value) {
    _seed = value;
}

void HDIParameters::set_minimum_gain(double value) {
    _minimum_gain = value;
}

void HDIParameters::set_eta(double value) {
    _eta = value;
}

void HDIParameters::set_momentum(double value) {
    _momentum = value;
}

void HDIParameters::set_final_momentum(double value) {
    _final_momentum = value;
}

void HDIParameters::set_mom_switching_iter(double value) {
    _mom_switching_iter = value;
}

void HDIParameters::set_exaggeration_factor(double value) {
    _exaggeration_factor = value;
}

void HDIParameters::set_remove_exaggeration_iter(unsigned int value) {
    _remove_exaggeration_iter = value;
}

void HDIParameters::set_theta(double value) {
    _theta = value;
}

HDIParameters & HDItSNE::parameters() {
    return _parameters;
}

void HDItSNE::run(unsigned int iterations) {
    hdi::dr::TSNE<double> tSNE;

    hdi::utils::CoutLog coutLog;
    tSNE.setLogger(&coutLog);

    tSNE.setDimensionality(_parameters._source_dimensions);

    for (unsigned int n = 0; n < _parameters._n_points; n++) {
        tSNE.addDataPoint(_parameters._source + n * _parameters._source_dimensions);
    }

    hdi::dr::TSNE<double>::InitParams params;
    params._perplexity = _parameters._perplexity;
    params._seed = _parameters._seed;
    params._embedding_dimensionality = _parameters._target_dimensions;
    params._minimum_gain = _parameters._minimum_gain;
    params._eta = _parameters._eta;
    params._momentum = _parameters._momentum;
    params._final_momentum = _parameters._final_momentum;
    params._mom_switching_iter = _parameters._mom_switching_iter;
    params._exaggeration_factor = _parameters._exaggeration_factor;
    params._remove_exaggeration_iter = static_cast<int>(_parameters._remove_exaggeration_iter);

    hdi::data::Embedding<double> embedding_container;
    tSNE.initialize(&embedding_container, params);

    if (_parameters._logger)
        _parameters._logger->log(HDILogger::INFO, "Computing gradient descent...");
    for (unsigned int iter = 0; iter < iterations; ++iter) {
        tSNE.doAnIteration();
        if (!(iter % 100) && _parameters._logger) {
            std::stringstream fmt;
            fmt << "iteration " << iter << "...";
            _parameters._logger->log(HDILogger::INFO, fmt.str());
        }
    }
    if (_parameters._logger)
        _parameters._logger->log(HDILogger::INFO, "... done");

    std::vector<double> &embedding = embedding_container.getContainer();
    std::copy(embedding.begin(), embedding.end(), _parameters._target);
}

HDIParameters & HDIaSNE::parameters() {
    return _parameters;
}

void HDIaSNE::run(unsigned int iterations) {
    std::vector<float> source_f(_parameters._source, _parameters._source + _parameters._n_points * _parameters._source_dimensions);

    hdi::dr::HDJointProbabilityGenerator<float>::Parameters prob_gen_param;
    prob_gen_param._perplexity = static_cast<float>(_parameters._perplexity);

    hdi::dr::HDJointProbabilityGenerator<float>::sparse_scalar_matrix_type distributions;

    hdi::dr::HDJointProbabilityGenerator<float> prob_gen;
    prob_gen.computeProbabilityDistributions(source_f.data(),
                                             _parameters._source_dimensions,
                                             _parameters._n_points,
                                             distributions,
                                             prob_gen_param);

    hdi::dr::SparseTSNEUserDefProbabilities<float>::Parameters tSNE_param;
    tSNE_param._embedding_dimensionality = static_cast<int>(_parameters._target_dimensions);
    tSNE_param._mom_switching_iter = _parameters._mom_switching_iter;
    tSNE_param._remove_exaggeration_iter = _parameters._remove_exaggeration_iter;

    hdi::data::Embedding<float> embedding_container;

    hdi::dr::SparseTSNEUserDefProbabilities<float> tSNE;
    tSNE.initialize(distributions, &embedding_container, tSNE_param);
    tSNE.setTheta(_parameters._theta);

    if (_parameters._logger)
        _parameters._logger->log(HDILogger::INFO, "Computing gradient descent...");
    for (unsigned int iter = 0; iter < iterations; ++iter) {
        tSNE.doAnIteration();
        if (!(iter % 100) && _parameters._logger) {
            std::stringstream fmt;
            fmt << "iteration " << iter << "...";
            _parameters._logger->log(HDILogger::INFO, fmt.str());
        }
    }
    if (_parameters._logger)
        _parameters._logger->log(HDILogger::INFO, "... done");

    std::vector<float> &embedding = embedding_container.getContainer();
    std::copy(embedding.begin(), embedding.end(), _parameters._target);
}
