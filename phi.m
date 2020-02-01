function y = phi(x)
  %% The sigmoid function used in all neurons.
  %% Its derivative needs to be contained in phiPrime.m
  
  y = 1 / (1 + exp(-x));
end
