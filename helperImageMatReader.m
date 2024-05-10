function data = helperImageMatReader(filename)
% helperImageMatReader Reads custom MAT files containing 5-channel 
% multispectral image data.
%
%  DATA = helperImageMatReader(FILENAME) returns the first 5 channels of the
%  multispectral image saved in FILENAME.

    d = load(filename);
    f = fields(d);
    data = d.(f{1})(:,:,1:5);
    index = isnan(data);
    data(index) = 0;
end
    
    
    