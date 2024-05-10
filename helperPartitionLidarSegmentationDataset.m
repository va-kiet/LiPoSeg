function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = helperPartitionLidarSegmentationDataset(imds, pxds, varargin)
% helperPartitionLidarSegmentationDataset   Partition Lidar data by randomly dividing data 
% for training and validation based on given ratio. Rest is used for 
% testing.
% 
%    'trainingDataPercentage'       - The trainingDataPercentage is used to divide the 
%                                input data into training set of given 
%                                ratio. Rest of the data is divided in 2:1
%                                ratio as validation and testing data.

p = inputParser;

defaultTrainingRatio = 0.7;
addOptional(p,'trainingDataPercentage', defaultTrainingRatio, @isfloat);

p.KeepUnmatched = true;
parse(p,varargin{:})

ratio = p.Results.trainingDataPercentage;

if(ratio >= 1)
    error('Ratio provided is incorrect. No testing data, please update ratio.');
elseif (ratio <= 0)
    error('Ratio provided is incorrect. Negative or zero given.');
end

% Set initial random state for example reproducibility.
rng(0);
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Using ratio percentage of images for training.
numTrain = round(ratio * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 33% of the rest of images for validation.
numVal = round((1-ratio)/3 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image ds for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages, 'FileExtensions', '.mat', 'ReadFcn', @helperImageMatReader);
imdsVal = imageDatastore(valImages, 'FileExtensions', '.mat', 'ReadFcn', @helperImageMatReader);
imdsTest = imageDatastore(testImages, 'FileExtensions', '.mat', 'ReadFcn', @helperImageMatReader);

% Extract class and label IDs info.
classNames = pxds.ClassNames;
labelIDs = 1:numel(classNames);

% Creating pixel label ds for training and test set.
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainingLabels, classNames, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classNames, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classNames, labelIDs);
end