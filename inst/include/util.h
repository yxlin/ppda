#include <armadillo> // Armadillo library
arma::vec getEdges(arma::vec z, double dt);
arma::vec getFilter(double m, double M, double h, double p); 
arma::vec pmax(arma::vec v, double min);
arma::vec getVec(double *x, int *nx);
arma::vec density(arma::vec y, arma::vec be, double dt);
double cquantile(arma::vec y, double q);
double bwNRD0(arma::vec y, double m);
double gaussian(double y, arma::vec yhat, double h);

