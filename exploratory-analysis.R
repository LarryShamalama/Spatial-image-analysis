library(geoR)

mnist <- read.csv('~/Documents/GitHub/Spatial-image-analysis/images/handwritten_digit.csv')



mnist.cbind <- matrix(0, nrow=28*28, ncol=3)
iter <- 1

for (i in 1:28){
  for (j in 1:28){
    
    # j before i, because it works better for some reason
    # maybe I didn't properly initialize the .csv file...
    mnist.cbind[iter,] <- c(j, i, mnist[i, j])
    iter <- iter + 1
  }
}

mnist.geo <- as.geodata(mnist.cbind)
plot(mnist.geo)
