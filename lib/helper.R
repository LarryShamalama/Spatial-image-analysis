create.adj.matrix <- function(xdim, ydim){
  
  position.matrix <- t(matrix(1:(xdim*ydim), nrow=ydim, ncol=xdim))
  adj.matrix <- matrix(0, nrow=(xdim*ydim), ncol=(xdim*ydim)) # very sparse
  
  for (i in 1:xdim){
    for (j in 1:ydim){
      pos <- position.matrix[i, j]
      
      for (delta.i in c(-1, 0, 1)){
        for (delta.j in c(-1, 0, 1)){
          
          count <- 0
          over  <- (((i + delta.i) < xdim) && (j + delta.j) < ydim)
          under <- (((i + delta.i) > 0) && (j + delta.j) > 0)
          
          if (over && under){
            
            neighbor <- position.matrix[i + delta.i, j + delta.j]

            #print(paste('Position ', pos, ', neighbor ', neighbor))
            if (pos != neighbor){
              count <- count + 1
              adj.matrix[pos, neighbor] <- 1
              adj.matrix[neighbor, pos] <- 1
            }
          }
          
        }
      }
    }
  }
  
  return (adj.matrix)
}

expit <- function(x){
  return (exp(x)/(1+exp(x)))
}

pad <- function(array, epsilon){
  output <- c()
  
  for (x in array){
    if (x == 0){
      output <- c(output, epsilon)
    }
    else if (x == 1){
      output <- c(output, 1-epsilon)
    }
    else{
      output <- c(output, x)
    }
  }
  
  return (output)
}
