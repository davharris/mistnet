#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
// [[Rcpp::export]]

NumericMatrix matrixMultiplyGrad(
  int n_out,
  NumericMatrix error_grad, 
  NumericMatrix input_act
){
  
  arma::mat grad = Rcpp::as<arma::mat>(error_grad);
  arma::mat input = Rcpp::as<arma::mat>(input_act);
  arma::mat out(grad.n_cols, input.n_cols);
  
  for(int i=0; i<n_out; i++){
    out.row(i) = (grad.col(i).t() * input);
  }
  return wrap(out.t());
}
