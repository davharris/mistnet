#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector sigmoidVector(NumericVector v) {
  return 1. / (1. + exp(-v));
}

// [[Rcpp::export]]
NumericMatrix sigmoidMatrix(NumericMatrix m) {
  NumericMatrix out(m.nrow(), m.ncol());
  for(int i = 0; i< m.ncol(); i++){
      out(_, i) = sigmoidVector(m(_, i));
    };
  return out;
}
