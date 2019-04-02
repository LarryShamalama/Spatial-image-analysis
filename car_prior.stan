data{
    int<lower=0> N; 
    int<lower=0> N_edges;
    
    int<lower=0> Y[N];
    vector[N] log_E;
    
    int<lower=1, upper=N> node1[N_edges];  // node1[i] adjacent to node2[i]
    int<lower=1, upper=N> node2[N_edges];  // and node1[i] < node2[i]
}
parameters {
    real beta0;
    
    vector[N] phi;   // spatial effect, mean of 0 and is normally distributed
    real<lower=0> tau_phi;
}
transformed parameters{
    real<lower=0> sigma_phi = inv(sqrt(tau_phi));
}
model {
    Y ~ poisson_log(log_E + beta0);
    
    target += -0.5 * dot_self(phi[node1] - phi[node2]); // added to log-density
  
    sum(phi) ~ normal(0, sigma_phi * sqrt(N)); // sum to 0 restriction and prior specification
                                       // stan takes input sigma, not sigma^2 or tau^2
    
    beta0 ~ normal(0, 5);
    tau_phi ~ gamma(1, 1);
}
