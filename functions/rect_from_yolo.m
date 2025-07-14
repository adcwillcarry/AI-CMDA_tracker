function rect = rect_from_yolo(label,a,b)
% RECT_FROM_YOLO - Convert YOLO format label to rectangle coordinates
%
% Syntax: rect = rect_from_yolo(label, a, b)
%
% Inputs:
%   label - 1x5 array containing YOLO format label [class, x_center, y_center, width, height]
%           where coordinates are normalized (0-1) relative to image dimensions
%   a     - Image height in pixels
%   b     - Image width in pixels
%
% Outputs:
%   rect  - 1x4 array containing rectangle coordinates [x_min, y_min, width, height]
%           in pixel coordinates suitable for image cropping
%
% Description:
%   This function converts YOLO format bounding box coordinates (normalized
%   center coordinates and dimensions) to standard rectangle format (top-left
%   corner coordinates and dimensions) in pixel units.
%
% YOLO Format:
%   - x_center, y_center: normalized center coordinates (0-1)
%   - width, height: normalized dimensions (0-1)
%
% Output Rectangle Format:
%   - x_min, y_min: top-left corner coordinates in pixels
%   - width, height: dimensions in pixels
%
% Example:
%   label = [0, 0.5, 0.5, 0.4, 0.6];  % Center object, 40% width, 60% height
%   rect = rect_from_yolo(label, 480, 640);  % For 640x480 image
%   % Returns: [128, 96, 256, 288] (approximate pixel coordinates)

% Extract image dimensions from input parameters
imageWidth = b;    % Image width in pixels
imageHeight = a;   % Image height in pixels

% Convert normalized YOLO coordinates to pixel coordinates
% Extract normalized center coordinates and dimensions from label
x_center = label(2) * imageWidth;   % Convert normalized x_center to pixels
y_center = label(3) * imageHeight;  % Convert normalized y_center to pixels
width = label(4) * imageWidth;      % Convert normalized width to pixels
height = label(5) * imageHeight;    % Convert normalized height to pixels

% Calculate top-left corner coordinates from center coordinates
% YOLO uses center-based coordinates, but rectangle format uses top-left corner
x_min = round(x_center - width / 2);   % Left edge coordinate
y_min = round(y_center - height / 2);  % Top edge coordinate
x_max = round(x_center + width / 2);   % Right edge coordinate (for reference)
y_max = round(y_center + height / 2);  % Bottom edge coordinate (for reference)

% Return rectangle in format suitable for image cropping: [x, y, width, height]
rect = [x_min, y_min, width, height];
end

