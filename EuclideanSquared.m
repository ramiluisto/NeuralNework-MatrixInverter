function error = EuclideanSquared(input, target)
  %% The error function, one half of the square of the euclidean norm,
  %% is used in the network. Note that the error propagation algorighms
  %% assume this to be the error function in their structure. Thus
  %% changing this function, which is used only in the reporting part
  %% of the main algorithm, will not change the behaviour of the network.
  
  squares = (input(:) - target(:)).**2;
  error = 0.5*sum(squares);
end
