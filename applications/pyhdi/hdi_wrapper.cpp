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

unsigned int n_points = 0;
unsigned int source_dimensions = 0;
unsigned int target_dimensions = 0;
double *source = nullptr;
double *target = nullptr;

PyHDIException::PyHDIException(const std::string &__arg) : runtime_error(__arg) {}

void set_n_points(unsigned int value) {
    n_points = value;
}

void set_input(double *xx, unsigned int n_input_points, unsigned int dimensions) {
    if (n_points == 0) {
        throw PyHDIException("call set_n_points() before set_input()");
    }
    if (n_input_points != n_points) {
        std::stringstream fmt;
        fmt << "n_points not equal to matrix n_input_points, " << n_points << " != " << n_input_points;
        throw PyHDIException(fmt.str());
    }
    source = xx;
    source_dimensions = dimensions;
}

void set_output(double *yy, unsigned int n_output_points, unsigned int dimensions) {
    if (n_points == 0) {
        throw PyHDIException("call set_n_points() before set_output()");
    }
    if (n_output_points != n_points) {
        std::stringstream fmt;
        fmt << "n_points not equal to matrix n_output_points, " << n_points << " != " << n_output_points;
        throw PyHDIException(fmt.str());
    }
    target = yy;
    target_dimensions = dimensions;
}

void run_tsne(
        double perplexity,
        int seed,
        double minimum_gain,
        double eta,
        double momentum,
        double final_momentum,
        double mom_switching_iter,
        double exaggeration_factor,
        unsigned int remove_exaggeration_iter,
        unsigned int iterations
) {
    hdi::dr::TSNE<double> tSNE;

    hdi::utils::CoutLog coutLog;
    tSNE.setLogger(&coutLog);

    tSNE.setDimensionality(source_dimensions);

    for (unsigned int n = 0; n < n_points; n++) {
        tSNE.addDataPoint(source + n * source_dimensions);
    }

    hdi::dr::TSNE<double>::InitParams params;
    params._perplexity = perplexity;
    params._seed = seed;
    params._embedding_dimensionality = target_dimensions;
    params._minimum_gain = minimum_gain;
    params._eta = eta;
    params._momentum = momentum;
    params._final_momentum = final_momentum;
    params._mom_switching_iter = mom_switching_iter;
    params._exaggeration_factor = exaggeration_factor;
    params._remove_exaggeration_iter = static_cast<int>(remove_exaggeration_iter);

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
    std::copy(embedding.begin(), embedding.end(), target);
}


void run_asne(double perplexity, double theta, unsigned int exaggeration_iter, unsigned int iterations) {
    std::vector<float> source_f(source, source + n_points * source_dimensions);

    hdi::dr::HDJointProbabilityGenerator<float>::Parameters prob_gen_param;
    prob_gen_param._perplexity = static_cast<float>(perplexity);

    hdi::dr::HDJointProbabilityGenerator<float>::sparse_scalar_matrix_type distributions;

    hdi::dr::HDJointProbabilityGenerator<float> prob_gen;
    prob_gen.computeProbabilityDistributions(source_f.data(),
                                             source_dimensions,
                                             n_points,
                                             distributions,
                                             prob_gen_param);

    hdi::dr::SparseTSNEUserDefProbabilities<float>::Parameters tSNE_param;
    tSNE_param._embedding_dimensionality = static_cast<int>(target_dimensions);
    tSNE_param._mom_switching_iter = exaggeration_iter;
    tSNE_param._remove_exaggeration_iter = exaggeration_iter;

    hdi::data::Embedding<float> embedding_container;

    hdi::dr::SparseTSNEUserDefProbabilities<float> tSNE;
    tSNE.initialize(distributions, &embedding_container, tSNE_param);
    tSNE.setTheta(theta);

    std::cout << "Computing gradient descent..." << std::endl;
    for (unsigned int iter = 0; iter < iterations; ++iter){
        tSNE.doAnIteration();
    }
    std::cout << "... done" << std::endl;

    std::vector<float> &embedding = embedding_container.getContainer();
    std::copy(embedding.begin(), embedding.end(), target);

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
