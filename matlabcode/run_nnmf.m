clear all;
clc;
inputDataSize = 10000;
trainDataSize = 8000;
basisNum = 5;
inputFileName = './imagenet/prob.mmap';
fprintf('Reading memmap file...');
inputMmap = memmapfile(inputFileName, 'Format','single');
inputDataDim = [inputDataSize, 1000];
rpca = false;
if rpca == true
    addpath('./exact_alm_rpca/');
    addpath('./exact_alm_rpca/PROPACK/');
end

%%get the indices of testing and training class
indiceFilename = 'house-dog-cat-boat.csv';
indiceCSV = csvread(indiceFilename,1,0);
indiceAll = indiceCSV(indiceCSV(:, 1) < inputDataSize, :);
indiceNum = length(unique(indiceAll(:, 1)));
indiceRelative = [(1:length(indiceAll)).', indiceAll(:,2)];
indicesTrain = indiceRelative((indiceAll(:,1) < trainDataSize),:);
indicesTest = indiceRelative((indiceAll(:,1) > trainDataSize) & (indiceAll(:,1) <= inputDataSize),:);
testClass = 93; %for convinience, select first class, which is 83
indicesTrainClass = indicesTrain(indicesTrain(:,2) == testClass);
indicesTestClass = indicesTest(indicesTest(:,2) == testClass);


%%reshape the activation map
reshapedArray = reshape(inputMmap.Data, inputDataDim);
reshapedArray = reshapedArray(indiceAll(:, 1), :, :, :);
%reshapedArray = reshape(reshapedArray, indiceNum , 256, 13*13);
%reshapedArray = reshape(reshapedArray, indiceNum , 256 * 13 * 13);
reshapedArray = reshapedArray.';
if rpca == true
    %use the parameter setting from network L-S method 
    density=sum(sum(reshapedArray))/(length(reshapedArray).^2);
    sigma=1/sqrt(density*length(reshapedArray));
    theta = 0.5;
    %perform L-S decomposition
    [lowrankArray sparseArray] = exact_alm_rpca(reshapedArray, sigma);
    lowrankArray = double(lowrankArray > theta);
    [W, H] = nnmf(lowrankArray(:, 1:length(indicesTrain)), basisNum);
else
    [W, H] = nnmf(reshapedArray(: , 1:length(indicesTrain)), basisNum);
end

%draw

figure(1);
plot(H(:, indicesTrainClass));
figure(3)
plot(H(:, indicesTrain(indicesTrain(:,2) == 105)));
figure(2);
plot((reshapedArray( : , indicesTestClass ) \ W).');

%%calculate the quantile array for the activation array
sortedArray = sort(reshape(permute(reshapedArray,[2,1,3]), 256, inputDataSize * 169), 2, 'descend');
quantile = sortedArray(:, 1:50:50700);
binaryArray = zeros(size(reshapedArray));
for i = 1:256
    binaryArray((i - 1)*169 + 1 : i*169,:) = reshapedArray((i - 1)*169 + 1: i*169,:) > quantile(i,5);   
end
if rpca == true
    %use the parameter setting from network L-S method 
    density=sum(sum(binaryArray))/(length(binaryArray).^2);
    sigma=1/sqrt(density*length(binaryArray));
    theta = 0.5;
    %perform L-S decomposition
    [lowrankArray sparseArray] = exact_alm_rpca(binaryArray, sigma);
    lowrankArray = double(lowrankArray > theta);
    [W2, H2] = nnmf(lowrankArray(:, 1:trainDataSize), basisNum);
else
    %NMF
    [W2, H2] = nnmf(binaryArray(:, 1:trainDataSize), basisNum);
end
%draw
figure(3);
plot(H2(:, indicesTrainClass));
figure(4);
plot((binaryArray(:, indicesTestClass) \ W2).');
