#' @export
row.selector = setRefClass(
  Class = "row.selector",
  fields = list(
    rows = "integer",
    shuffle = "logical",
    dataset.size = "integer",
    n.minibatch = "integer",
    completed.epochs = "integer",
    minibatch.ids = "integer",
    pointer = "integer"
  ),
  methods = list(
    select = function(){
      raw.ids = seq(pointer, pointer + n.minibatch - 1L)
      inside.bounds = raw.ids <= dataset.size
      minibatch.ids[inside.bounds] <<- rows[raw.ids[inside.bounds]]
      
      pointer <<- (pointer + n.minibatch - 1L) %% dataset.size + 1L
      
      if(any(raw.ids >= dataset.size)){
        completed.epochs <<- completed.epochs + 1L
        if(shuffle){
          rows <<- sample.int(dataset.size)
        }
        minibatch.ids[!inside.bounds] <<- rows[seq_len(pointer - 1)]
      }
    }
  )
)
