function out = helperAugmentData(inp)
% Apply random horizontal flipping.
out = cell(size(inp));
% Randomly flip the five-channel image and pixel labels horizontally.
I = inp{1};
sz = size(I);
tform = randomAffine2d("XReflection",true);
rout = affineOutputView(sz,tform,"BoundsStyle","centerOutput");
out{1} = imwarp(I,tform,"OutputView",rout);
out{2} = imwarp(inp{2},tform,"OutputView",rout);
end