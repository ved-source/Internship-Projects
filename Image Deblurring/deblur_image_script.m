% DeblurImage.m
% This script loads a blurred image and applies blind deconvolution

% Step 1: Load the blurred image
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'}, 'Select a Blurred Image');
if isequal(filename,0)
    disp('User canceled file selection');
    return;
end
imgPath = fullfile(pathname, filename);
blurred = im2double(imread(imgPath));

% Step 2: Convert to grayscale if needed
if size(blurred, 3) == 3
    blurredGray = rgb2gray(blurred);
else
    blurredGray = blurred;
end

% Step 3: Estimate PSF (Point Spread Function)
psfSize = 15;
psf = fspecial('motion', 20, 45);  % you can tweak length/angle

% Step 4: Deblur using Lucy-Richardson deconvolution
numIterations = 20;
deblurred = deconvlucy(blurredGray, psf, numIterations);

% Step 5: Display Results
figure;
subplot(1,2,1);
imshow(blurredGray);
title('Original Blurred Image');

subplot(1,2,2);
imshow(deblurred);
title('Deblurred Result');

% Optionally save
choice = questdlg('Do you want to save the deblurred image?', ...
	'Save Option', ...
	'Yes','No','Yes');
if strcmp(choice, 'Yes')
    [savefile, savepath] = uiputfile('deblurred_result.png');
    if savefile
        imwrite(deblurred, fullfile(savepath, savefile));
        disp(['Saved to: ', fullfile(savepath, savefile)]);
    end
end
