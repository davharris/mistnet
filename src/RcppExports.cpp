// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// rcpp_add_biases
NumericMatrix rcpp_add_biases(NumericMatrix m, NumericVector v);
RcppExport SEXP marsrover_rcpp_add_biases(SEXP mSEXP, SEXP vSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    NumericMatrix m = Rcpp::as<NumericMatrix >(mSEXP);
    NumericVector v = Rcpp::as<NumericVector >(vSEXP);
    NumericMatrix __result = rcpp_add_biases(m, v);
    return Rcpp::wrap(__result);
END_RCPP
}
