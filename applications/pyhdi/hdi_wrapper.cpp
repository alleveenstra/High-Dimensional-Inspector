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


PyHDIException::PyHDIException(const std::string &__arg) : runtime_error(__arg) {}

void HDI_Parameters::set_n_points(unsigned int value) {
    _n_points = value;
}

void HDI_Parameters::set_input(double *matrix, unsigned int n_input_points, unsigned int dimensions) {
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

void HDI_Parameters::set_output(double *matrix, unsigned int n_output_points, unsigned int dimensions) {
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

void HDI_Parameters::set_perplexity(double value) {
    _perplexity = value;
}

void HDI_Parameters::set_seed(int value) {
    _seed = value;
}

void HDI_Parameters::set_minimum_gain(double value) {
    _minimum_gain = value;
}

void HDI_Parameters::set_eta(double value) {
    _eta = value;
}

void HDI_Parameters::set_momentum(double value) {
    _momentum = value;
}

void HDI_Parameters::set_final_momentum(double value) {
    _final_momentum = value;
}

void HDI_Parameters::set_mom_switching_iter(double value) {
    _mom_switching_iter = value;
}

void HDI_Parameters::set_exaggeration_factor(double value) {
    _exaggeration_factor = value;
}

void HDI_Parameters::set_remove_exaggeration_iter(unsigned int value) {
    _remove_exaggeration_iter = value;
}

void HDI_Parameters::set_theta(double value) {
    _theta = value;
}

HDI_Parameters & HDI_tSNE::parameters() {
    return _parameters;
}

void HDI_tSNE::run(unsigned int iterations) {
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

    std::cout << "Computing gradient descent..." << std::endl;
    for(unsigned int iter = 0; iter < iterations; ++iter){
        tSNE.doAnIteration();
        if (iter % 100 == 0) {
            std::cout << "Iter " << iter << std::endl;
        }
    }
    std::cout << "... done" << std::endl;

    std::vector<double> &embedding = embedding_container.getContainer();
    std::copy(embedding.begin(), embedding.end(), _parameters._target);
}

HDI_Parameters & HDI_aSNE::parameters() {
    return _parameters;
}

void HDI_aSNE::run(unsigned int iterations) {
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

    std::cout << "Computing gradient descent..." << std::endl;
    for (unsigned int iter = 0; iter < iterations; ++iter){
        tSNE.doAnIteration();
    }
    std::cout << "... done" << std::endl;

    std::vector<float> &embedding = embedding_container.getContainer();
    std::copy(embedding.begin(), embedding.end(), _parameters._target);

}


//void run_hsne() {
//
//    int iterations = 1000;
//
//    hdi::utils::CoutLog log;
//
//    hdi::analytics::MultiscaleEmbedderSystem multiscale_embedder;
//    multiscale_embedder.setLogger(&log);
//    hdi::analytics::MultiscaleEmbedderSystem::panel_data_type& panel_data = multiscale_embedder.getPanelData();
//
//    ////////////////////////////////////////////////
//    ////////////////////////////////////////////////
//    ////////////////////////////////////////////////
//
//
//    std::vector<float> source_f(source, source + n_points * source_dimensions * sizeof(double));
//    std::vector<float> target_f(target, target + target_n * target_dimensions * sizeof(double));
//
//    //Input
//
//    {//initializing panel data
//        for(int j = 0; j < source_dimensions; ++j){
//            panel_data.addDimension(std::make_shared<hdi::data::EmptyData>(hdi::data::EmptyData()));
//        }
//        panel_data.initialize();
//        panel_data.reserve(n_points);
//        for (int n = 0; n < n_points; n++) {
//
//            std::vector<float> partial(source_f.data() + n * source_dimensions * sizeof(float), source_f.data() + (n+1) * source_dimensions * sizeof(float));
//
//            panel_data.addDataPoint(std::make_shared<hdi::data::EmptyData>(hdi::data::EmptyData()), partial);
//        }
//    }
//
////    if(parser.isSet(normalization_option)){
////        std::cout << "Applying a min-max normalization" << std::endl;
////        hdi::data::minMaxNormalization(panel_data);
////    }
//
//    hdi::analytics::MultiscaleEmbedderSystem::hsne_type::Parameters params;
//    params._seed = -1;
//    params._mcmcs_landmark_thresh = 1.5;
//    params._num_neighbors = 30;
//    params._aknn_num_trees = 4;
//    params._aknn_num_checks = 1024;
//    params._transition_matrix_prune_thresh = 1.5;
//    params._mcmcs_num_walks = 200;
//    params._num_walks_per_landmark = 200;
//
//    params._monte_carlo_sampling = true;
//    params._out_of_core_computation = true;
//
//    int num_scales = static_cast<int>(std::log10(n_points));
//    std::string name("HSNE_Analysis");
//
////    if(parser.isSet(beta_option)){
////        params._mcmcs_landmark_thresh = std::atof(parser.value(beta_option).toStdString().c_str());
////    }
////    if(parser.isSet(neigh_option)){
////        params._num_neighbors = std::atoi(parser.value(neigh_option).toStdString().c_str());
////    }
////    if(parser.isSet(trees_option)){
////        params._aknn_num_trees = std::atoi(parser.value(trees_option).toStdString().c_str());
////    }
////    if(parser.isSet(checks_option)){
////        params._aknn_num_checks = std::atoi(parser.value(checks_option).toStdString().c_str());
////    }
////    if(parser.isSet(prune_option)){
////        params._transition_matrix_prune_thresh = std::atof(parser.value(prune_option).toStdString().c_str());
////    }
////    if(parser.isSet(walks_landmark_option)){
////        params._mcmcs_num_walks = std::atoi(parser.value(walks_landmark_option).toStdString().c_str());
////    }
////    if(parser.isSet(walks_similarities_option)){
////        params._num_walks_per_landmark = std::atoi(parser.value(walks_similarities_option).toStdString().c_str());
////    }
////    if(parser.isSet(scale_option)){
////        num_scales = std::atoi(parser.value(scale_option).toStdString().c_str());
////    }
////    if(parser.isSet(name_option)){
////        name = parser.value(scale_option).toStdString();
////    }
//
//    std::cout << "Scales: " << num_scales << std::endl;
//
//    std::cout << "dimensions " <<  target_dimensions << std::endl;
//
//
//    std::cout << "Computing gradient descent..." << std::endl;
//    for(int iter = 0; iter < iterations; ++iter){
//        multiscale_embedder.doAnIterateOnAllEmbedder();
//        std::cout << "Iter " << iter << std::endl;
//    }
//    std::cout << "... done" << std::endl;
//
////    std::vector<float> &fooVec = embedding_container.getContainer();
//    //std::copy(fooVec.begin(), fooVec.end(), target);
//}
