
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix rectify(NumericMatrix m){
  for(int i = 0; i< m.ncol(); i++){
      m(_, i) = clamp(0, m(_, i), INFINITY);
    };
  return m;
}
  