function dataArray = GenerateTestData(
			 amount,
			 dimension,
			 magnitude,
			 treshold,
			 type
		       )
  %% This function generates pairs of invertible matrices
  %% and their inverses. The result is an dimension**2 times amount matrix
  %% which is saved in a file called type+'Data.mat'.
  %% Magnitude controls the maximum absolute value of entries and
  %% treshold limits the determinant of the matrix from below.

  
  n = dimension;
  output = [];
  
  sprintf("Starting to generate %f instances of training data.", amount)
  
  counter = 0;
  
  while counter < amount
    Progress = sprintf("Generator currently at %5.2f %%.", 100*counter/amount)
    A = RandomNxMmatrix(n,n,magnitude);
    if abs(det(A)) >= treshold
      B = inv(A);
      
      output(:,counter+1) = cat(1,A(:),B(:));
      counter = counter + 1;
    end
  end

  trainingDataArray = output;
  filename = strcat(type,'Data.mat')
  save(filename, 'trainingDataArray');
  dataArray = output;
  

end
  
