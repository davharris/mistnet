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
    },
    initialize = function(...){}
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
#' @details __. Following Senior et al. ("An empirical study of learning rates in deep neural networks for speech recognition"), 
#' the squared gradients are initialized at K instead of 0. By default, K == 0.1
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
    squared.grad = "matrix",
    K = "numeric"
  ),
  methods = list(
    computeDelta = function(gradient){
      squared.grad <<- squared.grad + gradient^2
      delta <<- -learning.rate / sqrt(squared.grad) * gradient
    },
    initialize = function(delta, learning.rate, K, ...){
      if(!missing(K)){
        K <<- K
      }else{
        if(length(.self$K) == 0){
          K <<- 0.1
        }
      }
      if(!missing(delta)){
        delta <<- delta
        squared.grad <<- matrix(
          .self$K,
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

#' adam updater
#' 
#' @description An updater with adaptive step sizes. Adam allows different 
#' weights to have different effective learning rates, depending on how 
#' much that parameter has moved so far and on how much it has moved recently 
#' in one consistent direction.
#'
#' @field a_0 initial step size; default is 0.01
#' @field annealing_rate controls the step size at time \code{t}. Step size is 
#'        \code{a[t] = a_0 / sqrt(1 - annealing_rate + t*annealing_rate)}.
#'        Default is 0.001.
#' @field b1 exponential decay rate for first moment estimate; default is 0.9
#' @field b2 exponential decay rate for second moment estimate; default is 0.999
#' @field e epsilon (prevents divide-by-zero errors); default is 1E-8
#' @field m first moment estimates; all zero by default at initialization
#' @field v second moment estimates; all zero by default at initialization
#' @field t timestep; zero by default at initialization
#' @field delta the delta matrix (see \code{updater})
#' @export adam.updater
#' @exportClass adam.updater
adam.updater = setRefClass(
  Class = "adam.updater",
  contains = "updater",
  fields = list(
    a_0 = "numeric",
    annealing_rate = "numeric",
    b1 = "numeric",
    b2 = "numeric",
    e = "numeric",
    m = "matrix",
    v = "matrix",
    t = "integer",
    delta = "matrix"
  ),
  methods = list(
    computeDelta = function(gradient){
      t <<- t + 1L
      g = gradient
      
      rate = a_0 / sqrt(1 - annealing_rate + t*annealing_rate)
      
      # Update biased moment estimates
      m <<- b1 * m + (1 - b1) * g
      v <<- b2 * v + (1 - b2) * g^2
      
      # Compute bias-corrected moment estimates
      m_hat = m / (1 - b1^t)
      v_hat = v / (1 - b2^t)
       
      delta <<- -rate * m_hat / (sqrt(v_hat) + e)
    },
    initialize = function(a_0 = 0.1, b1 = 0.9, b2 = 0.999, e = 1E-8,
                          t = 0L, delta, annealing_rate = .001, ...){
      if (length(.self$annealing_rate) == 0 | !missing(annealing_rate))  {annealing_rate <<- annealing_rate}
      if (length(.self$a_0) == 0 | !missing(a_0)) {a_0 <<- a_0}
      if (length(.self$b1)  == 0 | !missing(b1))  {b1 <<- b1}
      if (length(.self$b2)  == 0 | !missing(b2))  {b2 <<- b2}
      if (length(.self$e)   == 0 | !missing(e))   {e <<- e}
      if (length(.self$t)   == 0 | !missing(t))   {t <<- t}
      if (!missing(delta)) {
        delta <<- delta
        m <<- delta * 0
        v <<- delta * 0
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

#' rmsprop updater
#' 
#' @description Another updater with adaptive step sizes, like adagrad and adadelta.
#'
#' @details https://climin.readthedocs.org/en/latest/rmsprop.html
#'
#' @field learning.rate the learning rate (set to one in the original paper)
#' @field squared.grad a matrix summing the squared gradients over previous
#' updates (decays according to gamma)
#' @field decay how quickly should squared gradients decay?
#' @field delta the delta matrix (see \code{updater})
#' @export rmsprop.updater
#' @exportClass rmsprop.updater
rmsprop.updater = setRefClass(
  Class = "rmsprop.updater",
  contains = "updater",
  fields = list(
    delta = "matrix",
    learning.rate = "numeric",
    squared.grad = "matrix",
    decay = "numeric",
    leakage = "numeric"
  ),
  methods = list(
    computeDelta = function(gradient){
      squared.grad <<- squared.grad * (1 - decay + leakage) + decay * gradient^2
      delta <<- -learning.rate / sqrt(squared.grad + 1E-8) * gradient
    },
    initialize = function(delta, learning.rate, decay, ...){
      if(!missing(delta)){
        delta <<- delta
        squared.grad <<- matrix(
          0,
          nrow = nrow(delta),
          ncol = ncol(delta)
        )
      }
      if(!missing(learning.rate)){
        learning.rate <<- learning.rate
      }
      if(!missing(decay)){
        decay <<- decay
      }
      if(!missing(decay)){
        leakage <<- leakage
      }else{
        leakage <<- 0
      }
    }
  )
)