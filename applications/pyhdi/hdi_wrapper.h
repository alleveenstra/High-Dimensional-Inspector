class PyHDIException : public std::runtime_error
{
public:
    explicit PyHDIException(const std::string &);
};

void set_n_points(unsigned int value);
void set_input(double *matrix, unsigned int n_points, unsigned int dimensions);
void set_output(double *matrix, unsigned int n_points, unsigned int dimensions);
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