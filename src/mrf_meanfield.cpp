#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
// [[Rcpp::export]]

NumericMatrix mrf_meanfield(
  NumericMatrix rinput,
  NumericMatrix rlateral,
  int maxit,
  double damp,
  double tol
){
  arma::mat input = Rcpp::as<arma::mat>(rinput);
  arma::mat lateral = Rcpp::as<arma::mat>(rlateral);
  
  // Should factor out this sigmoid function, but am struggling with syntax
  arma::mat prob = 1 / (1 + exp(-input));

  
  for(int i=0; i<maxit; i++){
    arma::mat newprob = 1 / (1 + exp(-(prob * lateral + input)));
    
    // Dampening: only move (e.g.) 99% toward new value with damp == .01.
    // Prevents oscillations: Welling and Hinton: New algo for mean field
    //   Boltzmann Machines, ICANN 2002
    prob = newprob * (1 - damp) + prob * damp;
    
    if(mean(mean(abs(newprob - prob))) < tol){
      break;
    }
  }
  
  return wrap(prob);
}
