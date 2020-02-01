function [wDeltas, cDeltas] = CalculateDeltas(
				  testData,
				  networkState,
				  weightCollection,
				  constantCollection,
				  sizeArray )
     % This function will calculate the partial derivatives of a 
     % net given a datapoint [input, output], the interneural weights,
     % the constant weights and the topology of the net.

  
  depth = length(sizeArray);
  width = max(sizeArray);

  X = testData(:,1);
  Y = testData(:,2);
  
  ## We find first the derivatives w.r.t. the neuron inputs, i.e. del E / del z_j^L.
  ## Note that the column 1 will not be used, as we will not try to change the inputs.
  ## It's contained in the matrix for notational consistensy.
  networkDeltaMatrix = zeros(width,depth);

  ## Fill the output level derivatives.
  for j = 1:sizeArray(depth)
    networkDeltaMatrix(j,depth) = (networkState(j,depth) - Y(j));#*phiPrime(networkState(j,depth));
  end

  ## Backpropagate to fill the rest of the derivatives.
  for L = depth-1:-1:2
    n = sizeArray(L);
    m = sizeArray(L+1);
    
    weights = weightCollection(1:n,1:m,L);
    deltas = networkDeltaMatrix(1:m, L+1);
    zetas = networkState(1:n,L);


    ## This could be achieved with matrix multiplication trics, but it turned out to
    ## be unpleasant to troubleshoot.
    for j = 1:n
      sum = 0;
      for k = 1:m
	deltas(k)*weights(j,k)*phiPrime(zetas(j));
      end
      networkDeltaMatrix(j,L) = sum;
    end
    
  end

  ## From the previous deltas we can now easily calculate the partial derivatives
  ## w.r.t. the interneuronal weights and the constant weights.
  wDeltas = [];
  cDeltas = zeros(width,depth-1);
  for L = 1:depth-1
    n = sizeArray(L);
    m = sizeArray(L+1);
    
    wDeltas(1:n,1:m,L) = arrayfun(@phi, networkState(1:n,L))*networkDeltaMatrix(1:m,L+1)';
    cDeltas(1:m,L) = networkDeltaMatrix(1:m,L+1);
  end

  
end
