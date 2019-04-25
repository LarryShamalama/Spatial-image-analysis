rm(list=ls())
setwd('~/Documents/GitHub/Spatial-image-analysis/')

library(fields,quietly=TRUE)
colfunc <- colorRampPalette(c("blue","lightblue","white","yellow","orange","red"))

library(CARBayes)
library(spdep)

source('lib/helper.R')

# note that all data and csv files were obtained from Python
# specifically data-conversion.ipynb

adj.matrix <- create.adj.matrix(28, 28)

handwritten.digit <- read.csv('images/flatten_handwritten_digit.csv', header=FALSE)
handwritten.digit <- t(handwritten.digit)[1,]

#############################
### MORAN'S PLOT AND TEST ###
#############################

par(mfrow=c(1, 1))
moran <- moran.test(x=handwritten.digit,
                    listw=mat2listw(adj.matrix))
moran.plot(x=handwritten.digit,
           listw=mat2listw(adj.matrix),
           xlab='Handwritten digit',
           ylab='Spatially lagged handwritten digit',
           main='Moran\'s plot for handwritten digit')



handwritten.digit <- expit(handwritten.digit)

#######################
### DIGIT MODEL FIT ###
#######################


fit.12 <- S.CARleroux(handwritten.digit ~ 1,
                      family='gaussian',
                      burnin=1000,
                      prior.mean.beta=c(0),
                      prior.var.beta=c(25),
                      W=adj.matrix,
                      prior.nu2=c(1, 0.1),
                      prior.tau2=c(1, 0.1),
                      rho=1, # no hetero effect
                      n.sample=11000)

phi.samples <- fit.12$samples$phi
phi.means   <- as.numeric(as.matrix(colMeans(as.matrix(phi.samples))))
phi.means   <- matrix(phi.means, nrow=28)

par(mfrow=c(1, 1))

image.plot(1:28,
           1:28,
           phi.means,
           col=gray((0:255)/255),
           main=expression(paste('Mean of sampled spatial effects ', phi[i][j])),
           xlab='x coordinate',
           ylab='y coordinate')

phi.sd <- c()
for (i in 1:784){
  phi.sd <- c(phi.sd, sd(phi.samples[,i]))
}

colfunc <- colorRampPalette(c("black", "white")) 
image.plot(1:28,
           1:28,
           matrix(phi.sd, nrow=28, ncol=28),
           col=colfunc(100),
           main=expression(paste('Standard deviation of sampled spatial effects ', phi[i][j])),
           xlab='x coordinate',
           ylab='y coordinate')




par(mfrow=c(3, 1), mar=c(3, 2, 2, 2))

x.beta <- seq(0.3, 0.6, by=0.001)
beta.prior <- dnorm(x.beta, mean=0, sd=5)
plot(density(fit.12$samples$beta), 
     xlim=c(0.3, 0.6),
     main=expression(paste('Posterior sample of ', beta)))
lines(x.beta, beta.prior, col='red', lty=2)

x.nu2 <- seq(0, 0.02, by=0.00001)
nu2.prior <- 1/dgamma(x.nu2, shape=1, scale=1/0.1)
plot(density(fit.12$samples$nu2), 
     xlim=c(0, 0.02),
     main=expression(paste('Posterior sample of ', nu^2)))
lines(x.nu2, nu2.prior, col='red', lty=2)

x.tau2 <- seq(0, 0.02, by=0.00001)
tau2.prior <- 1/dgamma(x.tau2, shape=1, scale=1/0.1)
plot(density(fit.12$samples$tau2), 
     xlim=c(0, 0.02),
     main=expression(paste('Posterior sample of ', tau^2)))
lines(x.tau2, tau2.prior, col='red', lty=2)



#####################
### BREAST CANCER ###
#####################

# still takes way too long
# even though picture was scaled to 5% of its normal resolution

benign <- read.csv('images/benign.csv', header=FALSE)
x.dim.benign <- dim(benign)[1]
y.dim.benign <- dim(benign)[2]

adj.matrix.benign <- create.adj.matrix(x.dim.benign, y.dim.benign)

benign <- as.numeric(as.matrix(benign))

fit.benign <- S.CARleroux(expit(pad(benign, 1e-4)) ~ 1,
                          family='gaussian',
                          burnin=1000,
                          prior.mean.beta=c(0),
                          prior.var.beta=c(25),
                          W=adj.matrix.benign,
                          prior.nu2=c(1, 0.01),
                          prior.tau2=c(1, 0.01),
                          rho=1, # no hetero effect
                          n.sample=11000)

# insert model fit for malignant
# (didn't work for benign so didn't bother)

#####################
##### EXTENSION #####
#####################

# running on multiple images

adj.matrix <- create.adj.matrix(28, 28)
all.digits <- read.csv('images/all_digits.csv', header=FALSE)

N <- length(all.digits[,1])
parameter.means <- matrix(0, nrow=N, ncol=3)

# next bit takes a while
for (i in 1:N){
  # 10 first are 0's, then 10 next are 1's, etc.
  temp.fit <- S.CARleroux(expit(as.numeric(all.digits[i,])) ~ 1,
                          family='gaussian',
                          burnin=300,
                          prior.mean.beta=c(0),
                          prior.var.beta=c(25),
                          W=adj.matrix,
                          prior.nu2=c(1, 0.01),
                          prior.tau2=c(1, 0.01),
                          rho=1, # no hetero effect
                          n.sample=3300) # fewer samples should be okay...
  
  beta.mean <- mean(temp.fit$samples$beta)
  nu2.mean  <- mean(temp.fit$samples$nu2)
  tau2.mean <- mean(temp.fit$samples$tau2)
  
  parameter.means[i,] <- matrix(c(beta.mean,
                                nu2.mean,
                                tau2.mean),
                                ncol=1)
}

nice.plots <- function(result, col1, col2){
  '
  results should be parameters.means
  
  x axis: col1
  y axis: col2
  '
  
  cols <- c('#0000ff', # blue
            '#ffa500', # orange
            '#00a90a', # green
            '#ff0000', # red
            '#a020f0', # purple
            '#a0522d', # brown
            '#ff69b4', # pink
            '#696969', # grey
            '#6b8e23', # olive,
            '#00ffff') # cyan
  
  N <- length(result[,1])/10# number of samples for each digit
  
  minx <- min(result[,col1])
  maxx <- max(result[,col1])
  delta.x <- maxx - minx
  
  miny <- min(result[,col2])
  maxy <- max(result[,col2])
  delta.y <- maxy - miny
  
  xlim <- c(minx - delta.x*0.05, maxx + delta.x*0.05)
  ylim <- c(miny - delta.y*0.05, maxy + delta.y*0.05)
  
  plot(result[1:N, col1], 
       result[1:N, col2], 
       col=cols[1], 
       xlim=xlim,
       ylim=ylim,
       pch=20, # filling
       cex=1.2)  # dot size
  
  for (i in 2:10){
    par(new=FALSE)
    start <- 1 + N*(i-1)
    end   <- N*i
    points(result[start:end, col1], 
           result[start: end, col2], 
           col=cols[i], 
           xlim=xlim,
           ylim=ylim,
           pch=20,   # filling
           cex=1.2)  # dot size
  }
}

nice.plots(parameter.means, 1, 2) # titles and whatnot still needed
nice.plots(parameter.means, 2, 3)
nice.plots(parameter.means, 1, 3)

