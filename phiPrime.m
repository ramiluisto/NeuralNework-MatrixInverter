function y = phiPrime(x)
  %% The derivative of the sigmoid phi.m used in all neurons.
  
  y = phi(x)*(1-phi(x));
end


