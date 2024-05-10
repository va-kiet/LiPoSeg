function helperDisplayLidarOverlayImage(lidarImage,labelMap,classNames)
%  helperDisplayLidarOverlaidImage(lidarImage, labelMap, classNames)
%  displays the overlaid image. lidarImage is a five-channel lidar input.
%  labelMap contains pixel labels and classNames is an array of label
%  names.
% Read the intensity channel from the lidar image.
intensityChannel = uint8(lidarImage(:,:,4));
% Load the lidar color map.
cmap = helperPandasetColorMap;
% Overlay the labels over the intensity image.
B = labeloverlay(intensityChannel,labelMap,"Colormap",cmap,"Transparency",0.4);
% Resize for better visualization.
B = imresize(B,"Scale",[3 1],"method","nearest");
imshow(B);
helperPixelLabelColorbar(cmap,classNames);
end