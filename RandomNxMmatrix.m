function output = RandomNxMmatrix(N,
				  M,
				  rangeMax)
  %% A small macto to generate random matrices with given dimensions
  %% and a bound on the absolute values of the elemetns.
  
  output = 2*rangeMax*(rand(N,M) - 0.5*ones(N,M));
end
