
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix rcpp_add_biases(NumericMatrix m, NumericVector v) {
  NumericMatrix out(m.nrow(), m.ncol());
  for(int i = 0; i< m.ncol(); i++){
      // might be possible to speed this up with +=
      out(_, i) = m(_, i) + v[i];
    };
  return out;
}
