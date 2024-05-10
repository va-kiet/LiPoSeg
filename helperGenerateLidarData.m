function helperGenerateLidarData(lidarData, groundTruthData, imageDataLocation, labelDataLocation)
%helperGenerateLidarData Function to generate images and labels 
%   from Lidar format. 
%
%   generateLidarData(lidarPath,imageDataLocation,labelDataLocation) 
%   generates multi-channel images and labels given path. The inputs
%   lidarPath, imageDataLocation and labelDataLocation are described below.
% 
%  Inputs
%  ------
%   lidarData           Lidar point clouds as cell array.
%
%   groundTruthData     Ground truth data contains bounding box data as a
%                       time series for each class. The bounding box is 
%                       represented as a 9 dimensional array
%                       [x, y, z, length, width, height, yaw, pitch, roll]
%                       
% 
%   imageDataLocation   Folder where training images will be saved to
%                       disk. Make sure this points to a valid location on 
%                       the filesystem.
% 
%   labelDataLocation   Folder where training labels will be saved to
%                       disk. Make sure this points to a valid location on 
%                       the filesystem.

if ~exist(imageDataLocation,'dir')
    mkdir(imageDataLocation);
end

if ~exist(labelDataLocation,'dir')
    mkdir(labelDataLocation);
end

% Define class label values.
backgroundLabel = 1;
carLabel        = 2;
truckLabel      = 3;

% Load groundtruth timetable into workspace.
groundTruth = groundTruthData.bboxGroundTruth;
numFiles = size(groundTruth,1);

% Display progress on screen.
tmpStr = '';

for i=1:numFiles
    % Load ptcloud object.
    ptcloud = lidarData.lidarData{i};
    
    % Intialize label image with unknown values.
    labelImage = backgroundLabel*ones(size(ptcloud.Intensity));
    
    % Generate labels for points labeled as car.
    carColumnIndex = find(contains(groundTruth.Properties.VariableNames, 'car'), 1);
    if ~isempty(carColumnIndex)
        carMat = groundTruth(i,carColumnIndex).car{1};
        for j=1:size(carMat,1)
            carModel  = cuboidModel(carMat(j,:));
            withinBox = findPointsInsideCuboid(carModel,ptcloud);
            labelImage(withinBox) = carLabel;
        end
    end
    
    % Generate labels for points labeled as truck.
    truckColumnIndex = find(contains(groundTruth.Properties.VariableNames, 'truck'), 1);
    if ~isempty(truckColumnIndex)
        truckMat = groundTruth(i,truckColumnIndex).truck{1};
        for j=1:size(truckMat,1)
            truckModel = cuboidModel(truckMat(j,:));
            withinBox = findPointsInsideCuboid(truckModel,ptcloud);
            labelImage(withinBox) = truckLabel;            
        end
    end
    
    % Image are of 5-channels, namely x,y,z,intensity and range.
    im = helperPointCloudToImage(ptcloud);
    
    % Store images and labels as .mat and .png files respectively.
    imfile = fullfile(imageDataLocation,sprintf('%05d.mat',i));
    save(imfile,'im');
    lblfile = fullfile(labelDataLocation,sprintf('%05d.png',i));
    imwrite(uint8(labelImage),lblfile);
    
    % Display progress after 300 files on screen.
    if ~mod(i,300)
        msg = sprintf('Preprocessing data %3.2f%% complete', (i/numFiles)*100.0);
        fprintf(1,'%s',[tmpStr, msg]);
        tmpStr = repmat(sprintf('\b'), 1, length(msg));
    end
end

% Print completion message when done.
msg = sprintf('Preprocessing data 100%% complete');
fprintf(1,'%s',[tmpStr, msg]);

end