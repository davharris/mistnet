#' Updaters
#' 
#' @description Classes for determining the optimization step to make given a
#'  gradient
#' @details Updaters all have a \code{computeDelta} method, which determines the
#'  changes in coefficient values to make based on the estimated graidient and
#'  the current state of the updater.  These changes are stored in a field
#'  called \code{delta}.
#'  
#'  By default, the \code{mistnet} function uses the same parameters for all
#'  \code{updaters} in the network, but the user can tune them independently.
#' @export updater
#' @exportClass updater
updater = setRefClass(
  Class = "updater",
  fields = list(
    delta = "matrix"
  ),
  methods = list(
    computeDelta = function(...){
      stop("computeDelta not defined for this updater")
    }
  )
)

#' Stochastic gradient descent updater
#' 
#' @description An updater for descending a gradient with momentum
#'
#' @details __
#'
#' @field momentum the momentum term
#' @field learning.rate the learning rate
#' @field delta the delta matrix (see \code{updater})
#' @export sgd.updater
#' @exportClass sgd.updater
sgd.updater = setRefClass(
  Class = "sgd.updater",
  contains = "updater",
  fields = list(
    momentum = "numeric",
    learning.rate = "numeric",
    delta = "matrix"
  ),
  methods = list(
    initialize = function(delta, learning.rate, momentum){
      if(!missing(delta)){
        delta <<- delta
      }
      if(!missing(learning.rate)){
        learning.rate <<- learning.rate
      }
      if(!missing(momentum)){
        momentum <<- momentum
      }
    },
    computeDelta = function(gradient){
      delta <<- delta * momentum - gradient * learning.rate
    }
  )
)

#' adagrad updater
#' 
#' @description An updater with adaptive step sizes. Adagrad allows different 
#' weights to have different effective learning rates, depending on how 
#' much that parameter has moved so far.
#'
#' @details __
#'
#' @field learning.rate the learning rate (set to one in the original paper)
#' @field squared.grad a matrix summing the squared gradients over all previous
#'  updates
#' @field delta the delta matrix (see \code{updater})
#' @export adagrad.updater
#' @exportClass adagrad.updater
adagrad.updater = setRefClass(
  Class = "adagrad.updater",
  contains = "updater",
  fields = list(
    delta = "matrix",
    learning.rate = "numeric",
    squared.grad = "matrix"
  ),
  methods = list(
    computeDelta = function(gradient){
      squared.grad <<- squared.grad + gradient^2
      
      delta <<- -learning.rate / sqrt(squared.grad) * gradient
    },
    initialize = function(delta, learning.rate, ...){
      if(!missing(delta)){
        delta <<- delta
        # Don't initialize squared.grad at 0 to prevent divide by zero errors
        squared.grad <<- matrix(
          sqrt(.Machine$double.eps),
          nrow = nrow(delta),
          ncol = ncol(delta)
        )
      }
      if(!missing(learning.rate)){
        learning.rate <<- learning.rate
      }
    }
  )
)


# Epsilon is a fudge factor that determines initial rates and keeps things from
#    approaching zero.

#' adadelta updater
#' 
#' @description An updater with adaptive step sizes, like adagrad. 
#' Adadelta modifies adagrad (see \code{adagrad.updater}) by decaying the 
#' squared gradients and multiplying by an extra term to keep the units 
#' consistent.  Some evidence indicates that adadelta is more robust
#  to hyperparameter choices than adagrad or sgd.
#'
#' @details See Zeiler 2012
#' ADADELTA: AN ADAPTIVE LEARNING RATE METHOD
#' http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
#'
#' @field rho a rate (e.g. .95) that controls how long the updater "remembers" the 
#'  squared magnitude of previous updates.  Larger rho (closer to 1) allows the
#'  model to retain information from more steps in the past.
#' @field epsilon a small constant (e.g. 1E-6) to prevent numerical instability
#'  when dividing by small numbers
#' @field squared.grad a matrix summing the squared gradients over all previous
#'  updates, but decayed according to rho.
#' @field delta the delta matrix (see \code{updater})
#' @field squared.delta a matrix summing the squared deltas over all previous
#'  updates, but decayed according to rho.
#' @export adadelta.updater
#' @exportClass adadelta.updater
adadelta.updater = setRefClass(
  Class = "adadelta.updater",
  contains = "updater",
  fields = list(
    rho = "numeric",
    epsilon = "numeric",
    squared.grad = "matrix",
    delta = "matrix",
    squared.delta = "matrix"
  ),
  methods = list(
    initialize = function(delta, rho, epsilon){
      if(!missing(delta)){
        delta <<- delta
        squared.delta <<- delta
        squared.grad <<- delta
      }
      if(!missing(rho)){
        rho <<- rho
      }
      if(!missing(epsilon)){
        epsilon <<- epsilon
      }
    },
    RMS = function(x.squared){
      # Adding epsilon prevents division by tiny numbers
      sqrt(x.squared + epsilon)
    },
    computeDelta = function(gradient){
      # Line numbers correspond to Algorithm 1 in Zeiler 2012
      # ADADELTA: AN ADAPTIVE LEARNING RATE METHOD
      # http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
      
      # Line 4: accumulate gradient
      squared.grad <<- rho * squared.grad + (1 - rho) * gradient^2
      
      # Line 5: compute update. RMS(x) is calculated here as 
      #   `sqrt(x + epsilon)` to prevent zero values in the denominator.
      delta <<- -RMS(squared.delta) / RMS(squared.grad) * gradient
      
      # Line 6: accumulate updates
      squared.delta <<- rho * squared.delta + (1 - rho) * delta^2
    }
  )
)
