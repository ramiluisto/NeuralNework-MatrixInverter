## A neural network for inverting nxn matrices. 2020 Rami Luisto.
## 
## Currently it does not seem to converge properly, excepting with only two training
## data points which just means it learns to distinguish two n-vectors and return the
## n-vectors that are being trained, but even this is slow. The fact that it does
## converge in this case does seem to imply that the architecture itself is working,
## but something about the structure is not good for this problem... Maybe different
## sigmoids are in order?
##
## To operate, this file requires the following function files in the same folder:
## - phi.m
## - phiPrime.m
## - GenerateTestData.m
## - CalculateDeltas.m
## - EuclideanSquared.m
## - FillNetworkContent.m
## - RandomNxMmatrix.m

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



## Literal constants
## -Global
MATRIX_DIMENSION = 2;
WEIGHT_MATRIX_MAX_VALUE = 5;
## -Training cycle
TRAINING_ROUNDS = 1000;
BATCH_SIZE = 5;
## -Treining data
FORCE_NEW_TRAINING_DATA = true;
TRAINING_DATA_SIZE = 2;
TRAINING_MATRIX_MAX_VALUE = 5;
DETERMINANT_MIN_CUTOFF = 1;
## -Evolution
EPSILON = 0.1;
## -Testing cycle
TESTING_ROUNDS = 10;
## -Testing data
FORCE_NEW_TESTING_DATA = true;
TESTING_DATA_SIZE = TESTING_ROUNDS;
TESTING_MATRIX_MAX_VALUE = 5;
TEST_DETERMINANT_MIN_CUTOFF = 1;


## The matrix dimension defines the input and output layer sizes
ioLayerSize = MATRIX_DIMENSION**2;

########################################################################
## Modify the numbers in this array to alter the geometry of the net. ##
########################################################################
layerSizeArray = [ioLayerSize,6,6,ioLayerSize];


## Additional structural constants.
layerNo = length(layerSizeArray);
maximumLayerSize = max(layerSizeArray);






#####################################
## INITIALIZING NETWORK PARAMETERS ##
#####################################


## Creating the matrix collection hosting the interneural weights.
weightCollection = [];
for L = 1:layerNo-1
  weightCollection(:,:,L) = zeros(maximumLayerSize);

  n = layerSizeArray(L);
  m = layerSizeArray(L+1);
  weightCollection(1:n,1:m,L) = RandomNxMmatrix(n,m,WEIGHT_MATRIX_MAX_VALUE);
end

## Creating the matrix collection hosting the constant weights.
constantCollection = zeros(maximumLayerSize,layerNo-1);
for L = 1:layerNo-1
  n = layerSizeArray(L+1);
  constantCollection(1:n,L) = RandomNxMmatrix(n,1,WEIGHT_MATRIX_MAX_VALUE);
end

## Check if training data exists. If not, generate it.
sprintf("Checking if train data exists")
if exist('TrainingData.mat', 'file') != 2 | FORCE_NEW_TRAINING_DATA
  sprintf("Does not, generating...")
  
  GenerateTestData(
      TRAINING_DATA_SIZE,
      MATRIX_DIMENSION,
      TRAINING_MATRIX_MAX_VALUE,
      DETERMINANT_MIN_CUTOFF,
      'Training'
    );

  sprintf("Generated.")
end


## This will load an array called trainingDataArray
sprintf("Loading training data...")
load('TrainingData.mat')
sprintf("Ready.")

## Check if there is enough data points. If not, copy the training data.
while length(trainingDataArray) < TRAINING_ROUNDS*BATCH_SIZE
  sprintf("Not enough training data, expanding...")
  trainingDataArray = cat(2,trainingDataArray,trainingDataArray);
end



##############
## TRAINING ##
##############



## Initializing the vector that will contain all the current neuron states.
## Initializing here to make it persist outside of loops.
neuronContents = zeros(maximumLayerSize,layerNo);

## Initialize array to track error progression.
errorPlotter = [];

Status = sprintf("Starting the training...")
## Begin training the network
for round = 1:TRAINING_ROUNDS

  n = MATRIX_DIMENSION;
  
  ## Initialize the matrices that hold the deltas for the batch-averaging and the avg error calculator.
  weightDeltaAvg = zeros(size(weightCollection));
  constantDeltaAvg = zeros(size(constantCollection));
  errorAvg = 0;
  for batch = 1:BATCH_SIZE


    weighDeltas = [];
    constantDeltas = [];
    ## Extract input and expected output
    trainingDatum = trainingDataArray(:,round);
    input = trainingDatum(1:n**2);
    expected = trainingDatum(n**2+1:2*n**2);
    
    ## Fill the neuronContents
    neuronContents = FillNetworkContent(
			 input,
			 weightCollection,
			 constantCollection,
			 layerSizeArray
		       );
    
    ## Find the partial derivatives of the weights
    partialDerivatives = CalculateDeltas(
			     [input, expected],
			     neuronContents,
			     weightCollection,
			     constantCollection,
			     layerSizeArray
			   );
    weightDeltas = partialDerivatives(:,:,:,1);
    constantDeltas = partialDerivatives(1:maximumLayerSize,1:layerNo-1,2);

    ## Calculate error
    error = EuclideanSquared(neuronContents(1:ioLayerSize,layerNo), expected);

    ## Update the averaging variables
    weightDeltaAvg = weightDeltaAvg + (1/BATCH_SIZE)*weightDeltas;
    constantDeltaAvg = constantDeltaAvg + (1/BATCH_SIZE)*constantDeltas;
    errorAvg = errorAvg + (1/BATCH_SIZE)*error;
  end;
  
    
  ## Update the weights and constants
  weightCollection  = weightCollection - EPSILON.*weightDeltaAvg;
  constantCollection = constantCollection - EPSILON.*constantDeltaAvg;

  ## Update error array.
  errorPlotter(round) = errorAvg;
  
  ## Report progress
  Status = sprintf("|| Training: %6.1f %% | Error: %16.6f ||", 100*round/TRAINING_ROUNDS,error)
end
## Network training ends
Status = sprintf("Training complete.")





#############
## TESTING ##
#############

## Check if training data exists. If not, generate it.
sprintf("Checking if test data exists")
if exist('TestingData.mat', 'file') != 2 | FORCE_NEW_TESTING_DATA
  sprintf("Does not, generating...")
  
  GenerateTestData(
      TESTING_DATA_SIZE,
      MATRIX_DIMENSION,
      TESTING_MATRIX_MAX_VALUE,
      TEST_DETERMINANT_MIN_CUTOFF,
      'Testing'
    );

  sprintf("Generated.")
end

## This will load an array called testingDataArray
## The naming is a bit nonconsistent here, shoulld change how GenerateTestData.m works.
sprintf("Loading testing data...")
load('TestingData.mat');
sprintf("Ready.")

## Check if there is enough data points. If not, copy the training data.
while length(trainingDataArray) < TESTING_ROUNDS
  sprintf("Not enough testing data, expanding...")
  trainingDataArray = cat(2,trainingDataArray,trainingDataArray);
end



## Test the Network
for testround = 1:TESTING_ROUNDS

  n = MATRIX_DIMENSION;
  
  ## Extract input and expected output
  testingDatum = trainingDataArray(:,testround);
  input = testingDatum(1:n**2);
  expected = testingDatum(n**2+1:2*n**2);

  
  ## Fill the neuronContents
  neuronContents = FillNetworkContent(
		       input,
		       weightCollection,
		       constantCollection,
		       layerSizeArray
		     );

  ## Reprot progress
  Status = sprintf(
	       "|| Testing: %3.0f / %.0f | Error: %5.2f ||",
	       testround,
	       TESTING_ROUNDS,
	       EuclideanSquared( neuronContents(1:ioLayerSize,layerNo),input))
end

## Saving the matrices to actual files.
save InterNeuralWeights.mat weightCollection;
save ConstantNeuralWeights.mat constantCollection;

## Plot the error progression.
plot(1:TRAINING_ROUNDS, errorPlotter);
pause

