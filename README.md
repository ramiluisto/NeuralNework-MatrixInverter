# NeuralNework-MatrixInverter
Feed forward NN with variable topology for inverting matrices. Does not seem to converge atm.

For basic use, run MatrixInverterTrainerBatch.m and enjoy. For further use, the following is
the header of that file.


 A neural network for inverting nxn matrices. 2020 Rami Luisto.
 
 Currently it does not seem to converge properly, excepting with only two training
 data points which just means it learns to distinguish two n-vectors and return the
 n-vectors that are being trained, but even this is slow. The fact that it does
 converge in this case does seem to imply that the architecture itself is working,
 but something about the structure is not good for this problem... Maybe different
 sigmoids are in order?

 To operate, this file requires the following function files in the same folder:
 - phi.m
 - phiPrime.m
 - GenerateTestData.m
 - CalculateDeltas.m
 - EuclideanSquared.m
 - FillNetworkContent.m
 - RandomNxMmatrix.m

%{
This is a feed-forward network with input- and output layers the size
of n^2, where n is the MATRIX_DIMENSION. It learns in batches and updates
the weights with backpropagation and gradient descent. Topology
of the network can be altered by changing the layerSizeArray.
All neurons, excepting the output layer, use the sigmoid phi.m as their sigmoid. 
The output layer has the identity as its sigmoid.

The first list of literal constants can be altered to change the fundamental 
parameters. The training will do TRAINING_ROUNDS amount of batches, each of
which is the size of BATCH_SIZE. The constant EPSILON controls the jump
in the gradient descent. The determinant cutoffs control the singularity
of the test data, whereas the matrix max-values give upper bounds
to the absolute values of randomly generated matrices.

The testing and training data are saved into .mat -files and can be used
if present, but the creating of new train and test data can be forced
via the FORCE_NEW_*_DATA booleans.


In the end, the trained weight matrices are saved to InterNeuralWeights.mat
and ConstantNeuralWeights.mat and can be used idependently via

load(InterNeuralWeights.mat);
load(ConstantNeuralWeights.mat);
FillNetworkContent([~your input~], weightCollection, constantCollection, layerSizeArray)(:,layerNo)

to generate an output.

%}
