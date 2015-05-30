#' Prior distributions
#' 
#' Prior distributions regularize the model's weights during training
#' 
#' @export
#' @exportClass prior
prior = setRefClass(
  Class = "prior",
  fields = list(),
  methods = list(
    getLogGrad = function(x) stop({"log gradient not defined for this prior"}),
    sample = function(n){stop("sampler not defined for this prior")},
    update = function(...){stop("update not defined for this prior")}
  )
)

# gaussian.prior$sd can be a numeric matrix OR a numeric vector
setClassUnion("any.numeric", c("numeric", "matrix"))

#' @export gaussian.prior
#' @exportClass gaussian.prior
gaussian.prior = setRefClass(
  Class = "gaussian.prior",
  fields = list(
    mean = "numeric",
    sd = "any.numeric"
  ),
  contains = "prior",
  methods = list(
    getLogGrad = function(x){
      - (x - .self$mean) / .self$sd^2
    },
    sample = function(n){
      rnorm(n, mean = mean, sd = sd)
    },
    update = function(weights, update.mean, update.sd, min.sd){
      if(update.mean){
        mean <<- rowMeans(weights)
      }
      if(update.sd){
        var.vector = apply(weights, 1, var)
        var.vector = pmax(
          (var.vector + mean(var.vector)) / 2,
          min.sd^2
        )
        sd <<- replicate(ncol(weights), sqrt(var.vector))
      }
    }
  )
)

#' @export laplace.prior
#' @exportClass laplace.prior
laplace.prior = setRefClass(
  Class = "laplace.prior",
  fields = list(
    location = "numeric",
    scale = "numeric"
  ),
  contains = "prior",
  methods = list(
    getLogGrad = function(x){
      # This will work if the learning rate is small. Otherwise, it could 
      # overshoot past zero. That's probably not a big deal in practice?
      - sign(x - location) / scale
    },
    sample = function(n){
      rexp(n, rate  = 1 / scale) * base::sample(c(-1, 1), size = n, replace = TRUE)
    }
  )
) 


# GP prior assumes that covariance matrices (K) have already been determined
# by some other analysis.  Can re-initialize with new K & noise_sd after updating
# kernel hyperparameters as often as desired, though. Alternatively, one can just
# update the means, since that should be faster.

#' @export gp.prior
#' @exportClass gp.prior
gp.prior = setRefClass(
  Class = "gp.prior",
  fields = list(
    noise_sd = "numeric",
    means = "matrix",
    K = "array",
    L = "array",
    alpha = "array",
    v = "array",
    posterior_var = "array",
    inverse_var = "array"
  ),
  contains = "prior",
  methods = list(
    initialize = function(K, noise_sd, coefs){
      
      if(any(missing(K), missing(noise_sd), missing(coefs))){
        return()
      }
      
      # Notation roughly follows Rasmussen & Williams 2006
      # Gaussian Processes for Machine Learning
      # Algorithm 2.1, page 19
      assert_that(length(noise_sd) == dim(K)[3])
      
      noise_sd <<- noise_sd  # Noise SD is SD of data around the true function.
                             # Vector, with one element per covariance matrix.
      
      K <<- K                # K[ , , j] is the jth prior covariance matrix.
                             # Old comment says this should include noise along,
                             # the diagonal, but now I'm pretty sure that's incorrect?
      
      # Initialize L, v, posterior_var with correct dimensions
      L             <<- K * 0
      v             <<- K * 0
      posterior_var <<- K * 0
      
      I = diag(ncol(K)) # identity matrix, gets scaled by SD & added to diagonal of K
      
      # For each covariance matrix in K...
      for(i in seq_len(dim(K)[[3]])){
        # K matrix plus noise variance along the diagonal
        K_plus = K[ , , i] + I * noise_sd[i]^2
        
        # Cholesky decomposition is transposed to match algorithm in textbook
        L[ , , i] <<- t(chol(K_plus))
        
        # decomposition of variance explained by the data??
        v[ , , i] <<- solve(L[ , , i], K[ , , i])
        
        # posterior variation, used for sampling from posterior
        posterior_var[ , , i] <<- K_plus - crossprod(v[ , , i])
      }
      
      # Set prior mean function
      updateMeans(coefs)
      
    },
    getLogGrad = function(x){
      # Calculate gradient for each row of x independently.
      # sapply's output is transposed compared with what I want
      t_out = sapply(
        1:nrow(x),
        function(i){
          # https://stats.stackexchange.com/questions/90134/gradient-of-gaussian-log-likelihood
          # {1 \over 2}{\partial (\theta - \mu)^T\Sigma^{-1}(\theta - \mu) \over \partial \theta} = \Sigma^{-1}(\theta - \mu).
          - (x[i, ] - means[i, ]) / diag(posterior_var[,,i])
        }
      )
      
      t(t_out)
    },
    update = function(weights, update.mean, update.sd, min.sd){
      if(update.mean){
        updateMeans(weights)
      }
      if(update.sd){
        warning("no update.sd method implemented for gp priors")
      }
    },
    updateMeans = function(coefs, ...){
      # Each row of coefficients has its own posterior mean
      if(length(means) == 0){
        # Initialize means and alphas as NAs if blank
        means <<- matrix(NA, nrow = nrow(coefs), ncol = ncol(coefs))
        alpha <<- matrix(NA, nrow = nrow(coefs), ncol = ncol(coefs))
      }
      for(i in 1:nrow(coefs)){
        y = coefs[i, ]
        centered.y = y - mean(y)
        
        # Does this improperly pull species toward their current values, rather than just
        # toward their neighbors??
        
        # calculate new prior means on centered variables, then add the mean back on
        alpha[i, ] <<- solve(t(L[ , , i]), solve(L[ , , i], centered.y))
        means[i, ] <<- K[, , i] %*% alpha[i, ] + mean(y)
      }
    }
  )
)

