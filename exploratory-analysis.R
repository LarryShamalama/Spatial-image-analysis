library(geoR)

#####################
### ANALYSIS PLAN ###
#####################

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

# benign breast cancer radiography

benign <- read.csv('images/benign.csv', header=FALSE)
x.dim.benign <- dim(benign)[1]
y.dim.benign <- dim(benign)[2]

benign.cbind <- matrix(0, nrow=x.dim.benign*y.dim.benign, ncol=3)
iter <- 1

for (i in 1:x.dim.benign){
  for (j in 1:y.dim.benign){
    benign.cbind[iter,] <- c(i, j, benign[i, j])
    iter <- iter + 1
  }
  
}

benign.geo <- as.geodata(benign.cbind)
plot(benign.geo)

# malignant breast cancer radiography

malignant <- read.csv('images/malignant.csv', header=FALSE)
x.dim.malignant <- dim(malignant)[1]
y.dim.malignant <- dim(malignant)[2]

malignant.cbind <- matrix(0, nrow=x.dim.malignant*y.dim.malignant, ncol=3)
iter <- 1

for (i in 1:x.dim.malignant){
  for (j in 1:y.dim.malignant){
    malignant.cbind[iter,] <- c(i, j, malignant[i, j])
    iter <- iter + 1
  }
  
}

malignant.geo <- as.geodata(malignant.cbind)
plot(malignant.geo)
