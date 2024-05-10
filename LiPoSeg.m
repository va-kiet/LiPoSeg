lgraph = layerGraph();
tempLayers = imageInputLayer([64 1024 5],"Name","input");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv1_conv","Padding","same","Stride",[1 2])
    batchNormalizationLayer("Name","conv1_BN")
    reluLayer("Name","conv1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv_skip_conv","Padding","same")
    batchNormalizationLayer("Name","conv_skip_BN")
    reluLayer("Name","conv_skip_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","conv1_pool","Padding","same","Stride",[1 2])
    convolution2dLayer([1 1],16,"Name","fire1_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire1_squeeze1x1_BN")
    reluLayer("Name","fire1_squeeze1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","fire1_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire1_expand3x3_BN")
    reluLayer("Name","fire1_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","fire1_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire1_expand1x1_BN")
    reluLayer("Name","fire1_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire1_concat")
    convolution2dLayer([1 1],16,"Name","fire2_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire2_squeeze1x1_BN")
    reluLayer("Name","fire2_squeeze1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","fire2_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire2_expand3x3_BN")
    reluLayer("Name","fire2_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","fire2_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire2_expand1x1_BN")
    reluLayer("Name","fire2_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","fire2_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","CAM1_POOL","Padding","same","Stride",[2 2])
    groupedConvolution2dLayer([1 1],1,8,"Name","CAM1_conv","Padding","same")
    reluLayer("Name","CAM1_RELU")
    groupedConvolution2dLayer([1 1],16,8,"Name","CAM1_conv2","Padding","same")
    sigmoidLayer("Name","CAM1_SIG_2")
    resize2dLayer("Name","CAM1_resize","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","Scale",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","CAM1_REW");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","SR1_pool","Padding","same","Stride",[1 2])
    convolution2dLayer([1 1],32,"Name","fire3_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire3_squeeze1x1_BN")
    reluLayer("Name","fire3_squeeze1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","fire3_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire3_expand1x1_BN")
    reluLayer("Name","fire3_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","fire3_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire3_expand3x3_BN")
    reluLayer("Name","fire3_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire3_concat")
    convolution2dLayer([1 1],32,"Name","fire4_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire4_squeeze1x1_BN")
    reluLayer("Name","fire4_squeeze1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","fire4_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire4_expand1x1_BN")
    reluLayer("Name","fire4_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","fire4_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire4_expand3x3_BN")
    reluLayer("Name","fire4_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","fire4_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","CAM2_POOL","Padding","same","Stride",[2 2])
    groupedConvolution2dLayer([1 1],2,8,"Name","CAM2_conv","Padding","same")
    reluLayer("Name","CAM2_RELU")
    groupedConvolution2dLayer([1 1],32,8,"Name","CAM2_conv2","Padding","same")
    sigmoidLayer("Name","CAM2_SIG_2")
    resize2dLayer("Name","resize-scale","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","Scale",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","CAM2_REW");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","SR2_pool","Padding","same","Stride",[1 2])
    convolution2dLayer([1 1],48,"Name","fire5_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire5_squeeze1x1_BN")
    reluLayer("Name","fire5_squeeze1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","fire5_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire5_expand3x3_BN")
    reluLayer("Name","fire5_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","fire5_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire5_expand1x1_BN")
    reluLayer("Name","fire5_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire5_concat")
    convolution2dLayer([1 1],48,"Name","fire6_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire6_squeeze1x1_BN")
    reluLayer("Name","fire6_squeeze1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","fire6_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire6_expand1x1_BN")
    reluLayer("Name","fire6_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","fire6_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire6_expand3x3_BN")
    reluLayer("Name","fire6_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire6_concat")
    convolution2dLayer([1 1],64,"Name","fire7_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire7_squeeze1x1_BN")
    reluLayer("Name","fire7_squeeze1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","fire7_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire7_expand3x3_BN")
    reluLayer("Name","fire7_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","fire7_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire7_expand1x1_BN")
    reluLayer("Name","fire7_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire7_concat")
    convolution2dLayer([1 1],64,"Name","fire8_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire8_squeeze1x1_BN")
    reluLayer("Name","fire8_squeeze1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","fire8_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire8_expand3x3_BN")
    reluLayer("Name","fire8_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","fire8_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire8_expand1x1_BN")
    reluLayer("Name","fire8_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","fire8_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","CAM3_POOL","Padding","same","Stride",[2 2])
    groupedConvolution2dLayer([1 1],4,8,"Name","CAM3_conv","Padding","same")
    reluLayer("Name","CAM3_RELU")
    groupedConvolution2dLayer([1 1],64,8,"Name","CAM3_conv2","Padding","same")
    sigmoidLayer("Name","CAM3_SIG")
    resize2dLayer("Name","CAM3_RESIZE","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","Scale",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","SR3_REW");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([64 64],"Name","ASPP_GAP")
    groupedConvolution2dLayer([1 1],16,8,"Name","ASPP_GAP_conv","Padding","same")
    batchNormalizationLayer("Name","ASPP_GAP_BN")
    reluLayer("Name","ASPP_GAP_relu")
    resize2dLayer("Name","ASPP_GAP_resize","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[64 64])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([3 3],16,8,"Name","ASPP_1_conv","Padding","same")
    batchNormalizationLayer("Name","ASPP_1_BN")
    reluLayer("Name","ASPP_1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([3 3],16,8,"Name","ASPP_9_conv","DilationFactor",[9 9],"Padding","same")
    batchNormalizationLayer("Name","ASPP_9_BN")
    reluLayer("Name","ASPP_9_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([3 3],16,8,"Name","ASPP_6_conv","DilationFactor",[6 6],"Padding","same")
    batchNormalizationLayer("Name","ASPP_6_BN")
    reluLayer("Name","ASPP_6_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([3 3],16,8,"Name","ASPP_12_conv","DilationFactor",[12 12],"Padding","same")
    batchNormalizationLayer("Name","ASPP_12_BN")
    reluLayer("Name","ASPP_12_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(5,"Name","ASPP_concat")
    convolution2dLayer([1 1],128,"Name","ASPP_conv_conv","Padding","same")
    reluLayer("Name","ASPP_conv_relu")
    convolution2dLayer([1 1],32,"Name","fire_deconv_ASPP_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv_ASPP_squeeze1x1_BN")
    reluLayer("Name","fire_deconv_ASPP_squeeze1x1_relu")
    transposedConv2dLayer([1 4],32,"Name","fire_deconv_ASPP_deconv","Cropping","same","Stride",[1 2])
    reluLayer("Name","fire_deconv_ASPP_deconv_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","fire_deconv_ASPP_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv_ASPP_expand1x1_BN")
    reluLayer("Name","fire_deconv_ASPP_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","fire_deconv_ASPP_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv_ASPP_expand3x3_BN")
    reluLayer("Name","fire_deconv_ASPP_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","fire_deconv_ASPP_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","fire_deconv1_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv1_squeeze1x1_BN")
    reluLayer("Name","fire_deconv1_squeeze1x1_relu")
    transposedConv2dLayer([1 4],64,"Name","fire_deconv1_deconv","Cropping","same","Stride",[1 2])
    reluLayer("Name","fire_deconv1_deconv_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","fire_deconv1_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv1_expand1x1_BN")
    reluLayer("Name","fire_deconv1_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","fire_deconv1_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv1_expand3x3_BN")
    reluLayer("Name","fire_deconv1_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","fire_deconv1_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","fuse_fire_deconv1_concatSR2_REW");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","concat_fuse_fire_deconv1_concatSR2_REWfire_deconv_ASPP_concat")
    convolution2dLayer([1 1],64,"Name","fire_deconv2_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv2_squeeze1x1_BN")
    reluLayer("Name","fire_deconv2_squeeze1x1_relu")
    transposedConv2dLayer([1 4],64,"Name","fire_deconv2_deconv","Cropping","same","Stride",[1 2])
    reluLayer("Name","fire_deconv2_deconv_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","fire_deconv2_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv2_expand3x3_BN")
    reluLayer("Name","fire_deconv2_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","fire_deconv2_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv2_expand1x1_BN")
    reluLayer("Name","fire_deconv2_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","fire_deconv2_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","fuse_fire_deconv2_concatSR1_REW")
    convolution2dLayer([1 1],16,"Name","fire_deconv3_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv3_squeeze1x1_BN")
    reluLayer("Name","fire_deconv3_squeeze1x1_relu")
    transposedConv2dLayer([1 4],16,"Name","fire_deconv3_deconv","Cropping","same","Stride",[1 2])
    reluLayer("Name","fire_deconv3_deconv_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","fire_deconv3_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv3_expand1x1_BN")
    reluLayer("Name","fire_deconv3_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],32,"Name","fire_deconv3_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv3_expand3x3_BN")
    reluLayer("Name","fire_deconv3_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","fire_deconv3_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","fuse_fire_deconv3_concatconv1_relu")
    convolution2dLayer([1 1],16,"Name","fire_deconv4_squeeze1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv4_squeeze1x1_BN")
    reluLayer("Name","fire_deconv4_squeeze1x1_relu")
    transposedConv2dLayer([1 4],16,"Name","fire_deconv4_deconv","Cropping","same","Stride",[1 2])
    reluLayer("Name","fire_deconv4_deconv_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],32,"Name","fire_deconv4_expand3x3_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv4_expand3x3_BN")
    reluLayer("Name","fire_deconv4_expand3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","fire_deconv4_expand1x1_conv","Padding","same")
    batchNormalizationLayer("Name","fire_deconv4_expand1x1_BN")
    reluLayer("Name","fire_deconv4_expand1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","fire_deconv4_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","fuse_fire_deconv4_concatconv_skip_relu")
    dropoutLayer(0.5,"Name","dropout")
    convolution2dLayer([3 3],3,"Name","convlast_conv","Padding","same")
    reluLayer("Name","convlast_relu")
    softmaxLayer("Name","softmax")
    pixelClassificationLayer("Name","pixellabels")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

Connect Layer Branches
Connect all the branches of the network to create the network graph.
lgraph = connectLayers(lgraph,"input","conv1_conv");
lgraph = connectLayers(lgraph,"input","conv_skip_conv");
lgraph = connectLayers(lgraph,"conv1_relu","conv1_pool");
lgraph = connectLayers(lgraph,"conv1_relu","fuse_fire_deconv3_concatconv1_relu/in2");
lgraph = connectLayers(lgraph,"fire1_squeeze1x1_relu","fire1_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire1_squeeze1x1_relu","fire1_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire1_expand1x1_relu","fire1_concat/in1");
lgraph = connectLayers(lgraph,"fire1_expand3x3_relu","fire1_concat/in2");
lgraph = connectLayers(lgraph,"conv_skip_relu","fuse_fire_deconv4_concatconv_skip_relu/in2");
lgraph = connectLayers(lgraph,"fire2_squeeze1x1_relu","fire2_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire2_squeeze1x1_relu","fire2_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire2_expand3x3_relu","fire2_concat/in2");
lgraph = connectLayers(lgraph,"fire2_expand1x1_relu","fire2_concat/in1");
lgraph = connectLayers(lgraph,"fire2_concat","CAM1_POOL");
lgraph = connectLayers(lgraph,"fire2_concat","CAM1_REW/in2");
lgraph = connectLayers(lgraph,"CAM1_resize","CAM1_REW/in1");
lgraph = connectLayers(lgraph,"CAM1_REW","SR1_pool");
lgraph = connectLayers(lgraph,"CAM1_REW","fuse_fire_deconv2_concatSR1_REW/in2");
lgraph = connectLayers(lgraph,"fire3_squeeze1x1_relu","fire3_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire3_squeeze1x1_relu","fire3_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire3_expand1x1_relu","fire3_concat/in1");
lgraph = connectLayers(lgraph,"fire3_expand3x3_relu","fire3_concat/in2");
lgraph = connectLayers(lgraph,"fire4_squeeze1x1_relu","fire4_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire4_squeeze1x1_relu","fire4_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire4_expand3x3_relu","fire4_concat/in2");
lgraph = connectLayers(lgraph,"fire4_expand1x1_relu","fire4_concat/in1");
lgraph = connectLayers(lgraph,"fire4_concat","CAM2_POOL");
lgraph = connectLayers(lgraph,"fire4_concat","CAM2_REW/in2");
lgraph = connectLayers(lgraph,"resize-scale","CAM2_REW/in1");
lgraph = connectLayers(lgraph,"CAM2_REW","SR2_pool");
lgraph = connectLayers(lgraph,"CAM2_REW","fuse_fire_deconv1_concatSR2_REW/in2");
lgraph = connectLayers(lgraph,"fire5_squeeze1x1_relu","fire5_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire5_squeeze1x1_relu","fire5_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire5_expand3x3_relu","fire5_concat/in2");
lgraph = connectLayers(lgraph,"fire5_expand1x1_relu","fire5_concat/in1");
lgraph = connectLayers(lgraph,"fire6_squeeze1x1_relu","fire6_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire6_squeeze1x1_relu","fire6_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire6_expand1x1_relu","fire6_concat/in1");
lgraph = connectLayers(lgraph,"fire6_expand3x3_relu","fire6_concat/in2");
lgraph = connectLayers(lgraph,"fire7_squeeze1x1_relu","fire7_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire7_squeeze1x1_relu","fire7_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire7_expand1x1_relu","fire7_concat/in1");
lgraph = connectLayers(lgraph,"fire7_expand3x3_relu","fire7_concat/in2");
lgraph = connectLayers(lgraph,"fire8_squeeze1x1_relu","fire8_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire8_squeeze1x1_relu","fire8_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire8_expand3x3_relu","fire8_concat/in2");
lgraph = connectLayers(lgraph,"fire8_expand1x1_relu","fire8_concat/in1");
lgraph = connectLayers(lgraph,"fire8_concat","CAM3_POOL");
lgraph = connectLayers(lgraph,"fire8_concat","SR3_REW/in2");
lgraph = connectLayers(lgraph,"CAM3_RESIZE","SR3_REW/in1");
lgraph = connectLayers(lgraph,"SR3_REW","ASPP_GAP");
lgraph = connectLayers(lgraph,"SR3_REW","ASPP_1_conv");
lgraph = connectLayers(lgraph,"SR3_REW","ASPP_9_conv");
lgraph = connectLayers(lgraph,"SR3_REW","ASPP_6_conv");
lgraph = connectLayers(lgraph,"SR3_REW","ASPP_12_conv");
lgraph = connectLayers(lgraph,"SR3_REW","fire_deconv1_squeeze1x1_conv");
lgraph = connectLayers(lgraph,"ASPP_1_relu","ASPP_concat/in4");
lgraph = connectLayers(lgraph,"ASPP_9_relu","ASPP_concat/in1");
lgraph = connectLayers(lgraph,"ASPP_6_relu","ASPP_concat/in3");
lgraph = connectLayers(lgraph,"ASPP_GAP_resize","ASPP_concat/in2");
lgraph = connectLayers(lgraph,"ASPP_12_relu","ASPP_concat/in5");
lgraph = connectLayers(lgraph,"fire_deconv_ASPP_deconv_relu","fire_deconv_ASPP_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire_deconv_ASPP_deconv_relu","fire_deconv_ASPP_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire_deconv_ASPP_expand1x1_relu","fire_deconv_ASPP_concat/in1");
lgraph = connectLayers(lgraph,"fire_deconv_ASPP_expand3x3_relu","fire_deconv_ASPP_concat/in2");
lgraph = connectLayers(lgraph,"fire_deconv_ASPP_concat","concat_fuse_fire_deconv1_concatSR2_REWfire_deconv_ASPP_concat/in2");
lgraph = connectLayers(lgraph,"fire_deconv1_deconv_relu","fire_deconv1_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire_deconv1_deconv_relu","fire_deconv1_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire_deconv1_expand1x1_relu","fire_deconv1_concat/in1");
lgraph = connectLayers(lgraph,"fire_deconv1_expand3x3_relu","fire_deconv1_concat/in2");
lgraph = connectLayers(lgraph,"fire_deconv1_concat","fuse_fire_deconv1_concatSR2_REW/in1");
lgraph = connectLayers(lgraph,"fuse_fire_deconv1_concatSR2_REW","concat_fuse_fire_deconv1_concatSR2_REWfire_deconv_ASPP_concat/in1");
lgraph = connectLayers(lgraph,"fire_deconv2_deconv_relu","fire_deconv2_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire_deconv2_deconv_relu","fire_deconv2_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire_deconv2_expand3x3_relu","fire_deconv2_concat/in2");
lgraph = connectLayers(lgraph,"fire_deconv2_expand1x1_relu","fire_deconv2_concat/in1");
lgraph = connectLayers(lgraph,"fire_deconv2_concat","fuse_fire_deconv2_concatSR1_REW/in1");
lgraph = connectLayers(lgraph,"fire_deconv3_deconv_relu","fire_deconv3_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire_deconv3_deconv_relu","fire_deconv3_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire_deconv3_expand1x1_relu","fire_deconv3_concat/in1");
lgraph = connectLayers(lgraph,"fire_deconv3_expand3x3_relu","fire_deconv3_concat/in2");
lgraph = connectLayers(lgraph,"fire_deconv3_concat","fuse_fire_deconv3_concatconv1_relu/in1");
lgraph = connectLayers(lgraph,"fire_deconv4_deconv_relu","fire_deconv4_expand3x3_conv");
lgraph = connectLayers(lgraph,"fire_deconv4_deconv_relu","fire_deconv4_expand1x1_conv");
lgraph = connectLayers(lgraph,"fire_deconv4_expand1x1_relu","fire_deconv4_concat/in1");
lgraph = connectLayers(lgraph,"fire_deconv4_expand3x3_relu","fire_deconv4_concat/in2");
lgraph = connectLayers(lgraph,"fire_deconv4_concat","fuse_fire_deconv4_concatconv_skip_relu/in1");

plot(lgraph);
