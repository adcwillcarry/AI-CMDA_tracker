function img1= thdway(img1,threshold)
% THDWAY - Apply threshold-based binary segmentation to image
%
% Syntax: img1 = thdway(img1, threshold)
%
% Inputs:
%   img1      - Input image (grayscale or any numeric array)
%   threshold - Threshold value for binary segmentation
%
% Outputs:
%   img1      - Thresholded image where pixels <= threshold are set to 0,
%               pixels > threshold retain their original values
%
% Description:
%   This function performs threshold-based image segmentation by setting all
%   pixels with intensity values less than or equal to the threshold to zero,
%   while preserving the original values of pixels above the threshold.
%   This creates a binary mask where background (low intensity) regions are
%   suppressed and foreground (high intensity) regions are preserved.
%
% Algorithm:
%   - For each pixel: if pixel_value <= threshold, set to 0
%   - Otherwise: keep original pixel value
%
% Applications:
%   - Image preprocessing for object detection
%   - Noise reduction by removing low-intensity background
%   - Creating binary masks for region of interest
%   - Fluorescence microscopy image analysis
%
% Example:
%   img = imread('sample.jpg');
%   gray_img = rgb2gray(img);
%   threshold_val = 100;
%   segmented_img = thdway(double(gray_img), threshold_val);
%
% See also: Otsu, im2bw, imbinarize

% Apply threshold: set pixels <= threshold to zero, keep others unchanged
img1(img1<=threshold)=0;  % Vectorized operation for efficient thresholding
end