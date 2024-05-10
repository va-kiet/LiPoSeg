outputLocation = fullfile('output');
if ~exist(outputLocation,'dir')
    mkdir(outputLocation);
end
pxdsResults = semanticseg(imdsTest, net, ...
                'MiniBatchSize', 8, ...
                'WriteLocation', outputLocation, ...
                'Verbose', false);
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTest, 'Verbose', false);

metrics.DataSetMetrics
metrics.ClassMetrics