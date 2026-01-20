% Deblurring using Wiener Deconvolution in MATLAB
clc;
clear;

% Ask user to upload a blurred image
[filename, pathname] = uigetfile({'*.*','All Files (*.*)'}, 'Select a Blurred Image');
if isequal(filename,0)
    disp('User cancelled the upload.');
    return;
end

% Read the blurred image
blurredImage = im2double(imread(fullfile(pathname, filename)));
figure, imshow(blurredImage), title('Original Blurred Image');

% Estimate blur kernel (PSF - Point Spread Function)
psfSize = 9; % Size of the PSF
psfSigma = 2; % Standard deviation for Gaussian PSF
PSF = fspecial('gaussian', psfSize, psfSigma);

% Apply Wiener deconvolution
estimatedNoiseVar = 0.01;
deblurredImage = deconvwnr(blurredImage, PSF, estimatedNoiseVar);

% Show the deblurred image
figure, imshow(deblurredImage), title('Deblurred Image (Wiener Deconvolution)');