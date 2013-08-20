layer = setClass(
  Class = "layer",
  slots = list(
    coefficients = "matrix",
    state = "matrix",
    llik.grad.estimate = "matrix",
    grad.step = "matrix",
    learning.rate = "numeric",
    momentum = "numeric",
    regularize = "function"
  ),
  sealed = TRUE
)
