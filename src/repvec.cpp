
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix repvec(NumericVector v, double n){
  NumericMatrix m(v.size(), n);
  
  for(int i = 0; i< m.ncol(); i++){
    m(_, i) = v;
  };
  return m;
}
