library(geoR)

mnist <- read.csv('~/Documents/GitHub/Spatial-image-analysis/images/handwritten_digit.csv')
mnist <- t(handwritten.digit)[1,]
mnist <- matrix(handwritten.digit, nrow=28, ncol=28)


mnist.cbind <- matrix(0, nrow=28*28, ncol=3)
iter <- 1

for (i in 1:28){
  for (j in 1:28){
    
    mnist.cbind[iter,] <- c(i, j, mnist[i, j])
    iter <- iter + 1
  }
}

mnist.geo <- as.geodata(mnist.cbind)
plot(mnist.geo)
