outputFolder = fullfile('WPI');
imagesFolder = fullfile(outputFolder, 'images');
labelsFolder = fullfile(outputFolder, 'labels');
lidarData = load(fullfile(outputFolder, 'WPI_LidarData.mat'));
groundTruthData = load('WPI_LidarGroundTruth.mat');
helperGenerateTrainingData(lidarData, groundTruthData, imagesFolder, labelsFolder); 