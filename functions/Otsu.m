function ThreshValue = Otsu(Imag)
% OTSU - Calculate optimal threshold using Otsu's method
%
% Syntax: ThreshValue = Otsu(Imag)
%
% Inputs:
%   Imag - 2D array representing grayscale image intensities
%
% Outputs:
%   ThreshValue - Optimal threshold value calculated by Otsu's method
%
% Description:
%   This function implements Otsu's automatic thresholding algorithm, which
%   finds the optimal threshold by maximizing the between-class variance.
%   The method assumes the image contains two classes (foreground and background)
%   and finds the threshold that best separates these classes.
%
% Algorithm:
%   1. Calculate histogram of image intensities
%   2. For each possible threshold, compute between-class variance
%   3. Return threshold that maximizes between-class variance
%
% Example:
%   img = imread('image.jpg');
%   gray_img = rgb2gray(img);
%   threshold = Otsu(double(gray_img));
%   binary_img = gray_img > threshold;

% Find intensity range of the image
iMax = max(Imag(:));              % Maximum intensity value
iMin = min(Imag(:));              % Minimum intensity value
T = iMin:iMax;                    % Range of possible threshold values
Tval = zeros(size(T));            % Array to store variance values for each threshold
[iRow, iCol] = size(Imag);        % Get image dimensions
imagSize = iRow*iCol;             % Total number of pixels

% Iterate through all possible threshold values to find optimal one
for i = 1 : length(T)
    TK = T(i);                    % Current threshold being tested
    iFg = 0;                      % Foreground pixel count
    iBg = 0;                      % Background pixel count
    FgSum = 0;                    % Sum of foreground pixel intensities
    BgSum = 0;                    % Sum of background pixel intensities
    
    % Classify each pixel as foreground or background based on threshold
    for j = 1 : iRow
        for k = 1 : iCol
            temp = Imag(j, k);    % Current pixel intensity
            if temp > TK
                iFg = iFg + 1;          % Count foreground pixels
                FgSum = FgSum + temp;   % Sum foreground intensities
            else
                iBg = iBg + 1;          % Count background pixels
                BgSum = BgSum + temp;   % Sum background intensities
            end
        end
    end
    
    % Calculate class probabilities and mean intensities
    w0 = iFg/imagSize;            % Foreground class probability
    w1 = iBg/imagSize;            % Background class probability
    u0 = FgSum/iFg;               % Mean intensity of foreground class
    u1 = BgSum/iBg;               % Mean intensity of background class
    
    % Calculate between-class variance (Otsu's criterion)
    % Between-class variance = w0 * w1 * (u0 - u1)^2
    Tval(i) = w0*w1*(u0 - u1)*(u0 - u1);
end

% Find threshold that maximizes between-class variance
[~, flag] = max(Tval);            % Get index of maximum variance
ThreshValue = T(flag);            % Return corresponding threshold value
end