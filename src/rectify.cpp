
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix rectify(NumericMatrix m){
  NumericMatrix out(m.nrow(), m.ncol());
  for(int i = 0; i< m.ncol(); i++){
      out(_, i) = clamp(0, m(_, i), INFINITY);
    };
  return out;
}
  