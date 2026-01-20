clc;
clear;

% Load the uploaded image
blurredImage = im2double(imread('a4a7006b-4037-423e-8f54-dd9e1a12ef1f.png'));
figure, imshow(blurredImage), title('Original Blurred Image');

% Estimate the PSF (Gaussian blur assumption)
psfSize = 11;
psfSigma = 2;
PSF = fspecial('gaussian', psfSize, psfSigma);

% Apply Lucy-Richardson deconvolution
numIterations = 30;
deblurredImage = deconvlucy(blurredImage, PSF, numIterations);

% Show the result
figure, imshow(deblurredImage), title('Deblurred Image (Lucy-Richardson)');

% Optional: Save the result
imwrite(deblurredImage, 'deblurred_lucy_richardson.png');
disp('Deblurred image saved as "deblurred_lucy_richardson.png".');
