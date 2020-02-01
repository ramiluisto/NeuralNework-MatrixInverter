function neuronContents = FillNetworkContent(
				   input,
				   weightCollection,
				   constantCollection,
				   sizeArray
			    )
  %% This function is the "Neural Network Mapping", in the sense that
  %% given the current state and input, it calculates the internal values
  %% of all the neurons. (Internal values in the sese of weighted sums of the
  %% outputs of the previous layers; to get the outputs of hidden layers you
  %% need to apply the phi.m to the contents.) The last layer has the
  %% identity function as the treshold, so neuronContents(:,depth) will contain
  %% the output of the neural net.
  
  depth = length(sizeArray);
  width = max(sizeArray);
  ioSize = sizeArray(1);
  
  ## Initializing future output arrays and importing the input vector.
  contents = zeros(width, depth);
  contents(1:ioSize,1) =  input(:);

  outputs = zeros(width, depth);
  outputs(1:ioSize,1) =  arrayfun(@phi, contents(1:ioSize,1));
  

  ## Fill the rest of the array.
  for L = 2:depth
    n = sizeArray(L-1);
    m = sizeArray(L);

    ## Calculate contents of the next layer.
    contents(1:m,L) = weightCollection(1:n,1:m,L-1)'*outputs(1:n,L-1) + constantCollection(1:m,L-1);
    outputs(1:m,L) = arrayfun(@phi, contents(1:m,L) );
  end

  neuronContents = contents;
end
