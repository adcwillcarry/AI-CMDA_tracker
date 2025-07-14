%% Cell Division Metabolism Analysis Script
% This script analyzes cellular metabolism during division by calculating
% fluorescence intensity ratios between CFP and GFP channels
% Author: [Your Name]
% Date: [Current Date]

clear; clc; close all;
addpath('functions/')

% Initialize directory paths for images and labels
img_dir = '';  % Directory containing microscopy images
img_dir = strrep([img_dir,'/'],'\','/');
txt_dir = '';  % Directory containing YOLO format label files
txt_files = get_files(txt_dir, '*.txt');

% Load class definitions and identify mother-daughter relationships
classes = load('');  % Load class IDs from file
mother_index = [];   % Array to store [mother_id, daughter1_id, daughter2_id] triplets

% Find mother cells that have corresponding daughter cells
for i = 1:size(classes,1)
    if ismember(classes(i)*10, classes)
        mother_index = [mother_index; i-1, find(classes == classes(i)*10)-1, find(classes == classes(i)*10+1)-1];
    end
end

%% Main Analysis Loop - Process each mother-daughter triplet
figure;
for index = 1:size(mother_index,1)
    % Initialize arrays to store bounding boxes and frame indices
    m_rect = [];    % Mother cell rectangles
    d1_rect = [];   % Daughter 1 cell rectangles  
    d2_rect = [];   % Daughter 2 cell rectangles
    m_index = [];   % Frame indices where mother cell appears
    d1_index = [];  % Frame indices where daughter 1 appears
    d2_index = [];  % Frame indices where daughter 2 appears
    
    % Extract bounding boxes for mother and daughter cells across all frames
    for i = 1:size(txt_files,1)-1
        labels = load([txt_dir, txt_files{i}]);
        
        % Find mother cell in current frame
        if ismember(mother_index(index,1), labels(:,1))
            m_index = [m_index; i];
            m_rect = [m_rect; rect_shrink(rect_from_yolo(labels(labels(:,1)==mother_index(index,1),:), 904, 1224), 1)];
        end
        
        % Find daughter 1 cell in current frame
        if ismember(mother_index(index,2), labels(:,1))
            d1_index = [d1_index; i];
            d1_rect = [d1_rect; rect_shrink(rect_from_yolo(labels(labels(:,1)==mother_index(index,2),:), 904, 1224), 1)];
        end
        
        % Find daughter 2 cell in current frame
        if ismember(mother_index(index,3), labels(:,1))
            d2_index = [d2_index; i];
            d2_rect = [d2_rect; rect_shrink(rect_from_yolo(labels(labels(:,1)==mother_index(index,3),:), 904, 1224), 1)];
        end
    end
    
    % Array to store average fluorescence ratios [ratio, cell_type]
    % cell_type: 0 = mother cell, 1 = daughter cells
    average = [];
    %% Process Mother Cells - Calculate CFP/GFP ratio for each frame
    for i = 1:size(m_rect,1)
        % Parse filename to construct CFP and GFP image paths
        splits = split(txt_files{m_index(i)}, '_');
        cfp_name = [splits{1},'_',splits{2},'_1_',splits{4},'_CFP YFP FRET V2_',splits{6}];
        gfp_name = [splits{1},'_',splits{2},'_2_',splits{4},'_GFP_',splits{6}];
        
        % Load and process CFP channel image
        img_cfp = im2gray(im2uint8(imread([img_dir, strrep(cfp_name, 'txt', 'tif')])));
        img_gfp = im2gray(im2uint8(imread([img_dir, strrep(gfp_name, 'txt', 'tif')])));
        
        % Crop images to mother cell region and apply thresholding
        img_cfp = imcrop(img_cfp, m_rect(i,:));
        img_cfp = thdway(img_cfp, floor(Otsu(double(img_cfp))));
        img_gfp = imcrop(img_gfp, m_rect(i,:));
        img_gfp = thdway(img_gfp, floor(Otsu(double(img_cfp))));
        
        % Calculate CFP/GFP ratio and handle division by zero
        img_d = double(img_cfp) ./ double(img_gfp);
        img_d(img_d==0)=nan;  % Set zero values to NaN
        notNanValues = img_d(~isnan(img_d));
        notNanValues = notNanValues(~isinf(notNanValues));
        
        % Store average ratio for mother cell (type 0)
        average = [average; mean(notNanValues), 0];
        scatter(i, mean(notNanValues), 'r');  % Plot in red for mother cells
        hold on;
    end
    %% Process Daughter Cells - Calculate combined CFP/GFP ratio
    for i = 1:min(size(d1_rect,1), size(d2_rect,1))
        % Combine bounding boxes for both daughter cells
        labels_add = [floor(d1_rect(i,:)); floor(d2_rect(i,:))];
        
        % Determine which frame index to use for filename parsing
        if size(d1_rect,1) > size(d2_rect,1)
            splits = split(txt_files{d2_index(i)}, '_');
        else
            splits = split(txt_files{d1_index(i)}, '_');
        end
        
        % Construct image filenames
        cfp_name = [splits{1},'_',splits{2},'_1_',splits{4},'_CFP YFP FRET V2_',splits{6}];
        gfp_name = [splits{1},'_',splits{2},'_2_',splits{4},'_GFP_',splits{6}];
        
        % Load images
        img_gfp = im2gray(im2uint8(imread([img_dir, strrep(gfp_name, 'txt', 'tif')])));
        img_cfp = im2gray(im2uint8(imread([img_dir, strrep(cfp_name, 'txt', 'tif')])));
        
        % Create binary mask covering both daughter cell regions
        mask = zeros(size(img_gfp));
        
        % Ensure bounding box coordinates are within image bounds
        % Process daughter cell 1 region
        if 0 >= labels_add(1,1)
            labels_add(1,1) = 1;
        end
        row1 = labels_add(1,1) + labels_add(1,3);
        if row1 > size(img_gfp,2)
            row1 = size(img_gfp,2);
        end
        if 0 >= labels_add(1,2)
            labels_add(1,2) = 1;
        end
        col1 = labels_add(1,2) + labels_add(1,4);
        if col1 > size(img_gfp,1)
            col1 = size(img_gfp,1);
        end
        
        % Process daughter cell 2 region
        if 0 >= labels_add(2,1)
            labels_add(2,1) = 1;
        end
        row2 = labels_add(2,1) + labels_add(2,3);
        if row2 > size(img_gfp,2)
            row2 = size(img_gfp,2);
        end
        if 0 >= labels_add(2,2)
            labels_add(2,2) = 1;
        end
        col2 = labels_add(2,2) + labels_add(2,4);
        if col2 > size(img_gfp,1)
            col2 = size(img_gfp,1);
        end
        
        % Fill mask for both daughter cell regions
        for row = labels_add(1,1):row1
            for col = labels_add(1,2):col1
                mask(col, row) = 1;
            end
        end
        for row = labels_add(2,1):row2
            for col = labels_add(2,2):col2
                mask(col, row) = 1;
            end
        end
        % Calculate optimal thresholds for each daughter cell region
        img_cfp_crop = imcrop(img_cfp, labels_add(1,:));
        t1 = int8(Otsu(double(img_cfp_crop)));
        img_cfp_crop = imcrop(img_cfp, labels_add(2,:));
        t3 = int8(Otsu(double(img_cfp_crop)));
        img_gfp_crop = imcrop(img_gfp, labels_add(2,:));
        t4 = int8(Otsu(double(img_gfp_crop)));
        img_gfp_crop = imcrop(img_gfp, labels_add(1,:));
        t2 = int8(Otsu(double(img_gfp_crop)));
        
        % Apply mask and thresholding to GFP image
        img_gfp = double(img_gfp) .* double(mask);
        if isempty(max(max(t2, t4)))
            img_gfp = thdway(img_gfp, 0);
        else
            img_gfp = thdway(img_gfp, max(max(t2, t4)));
        end
        
        % Apply mask and thresholding to CFP image
        img_cfp = double(img_cfp) .* double(mask);
        if ~isempty(max(max(t3, t1)))
            img_cfp = thdway(img_cfp, max(max(t3, t1)));
        else
            img_cfp = thdway(img_cfp, 0);
        end
        
        % Calculate CFP/GFP ratio for daughter cells
        img_d = double(img_cfp) ./ double(img_gfp);
        img_d(img_d==0) = nan;  % Handle division by zero
        notNanValues = img_d(~isnan(img_d));
        notNanValues = notNanValues(~isinf(notNanValues));
        
        % Store average ratio for daughter cells (type 1)
        average = [average; mean(notNanValues), 1];
        scatter(size(m_rect,1)+i, mean(notNanValues), 'b');  % Plot in blue for daughter cells
    end
    
    % Interpolate missing values (NaN) using neighboring values
    for avr_index = 2:size(average,1)-1
        if isnan(average(avr_index,1))
            average(avr_index,1) = (average(avr_index-1,1) + average(avr_index+1,1)) / 2;
        end
    end
    
    % Format and save individual cell analysis results
    hold off;
    ylim([0.8, 1.8]);  % Set y-axis limits for fluorescence ratio
    folder = ['./analysis_labels/'];
    if exist(folder)==0
        mkdir(folder);  % Create output directory if it doesn't exist
    end
    if size(average,1) > 0
        % Save plot and data for current cell lineage
        exportgraphics(gcf, ['./analysis_labels/','cell',num2str(index),'.png'], 'Resolution', 300);
        dlmwrite(['./analysis_labels/','cell',num2str(index),'.txt'], average, 'delimiter', '\t');
    end
end
%% Combined Analysis - Generate Summary Plots
% Read processed data files and create comprehensive analysis plots
txt_dir = folder;
txt_dir = strrep([txt_dir,'/'],'\','/');
txt_files = get_files(txt_dir,'*.txt');

% Initialize figure for time-series analysis
figure;
hold on;
numPlots = 20;
colors = turbo(numPlots);  % Generate distinct colors for each cell lineage
min_max_list = [];         % Store min/max values for normalization
mean_divide = [];          % Store mean values for mitotic vs non-mitotic phases

% Define time window around division event
divide_x_min = -3;  % Frames before division
divide_x_max = 2;   % Frames after division

% Process each cell lineage data file
for i = 1:size(txt_files,1)
    f_lists = load([txt_dir,txt_files{i}]);
    if size(f_lists,2)>1
        % Find division time point (transition from mother to daughter cells)
        for ii = 1:size(f_lists,1)-1
            if f_lists(ii,2)~=f_lists(ii+1,2)
                index = ii;  % Division occurs between frame ii and ii+1
            end
        end
        
        % Create relative time scale centered on division event
        add_list = zeros(size(f_lists,1),1);
        add_list(index) = -1;  % Frame before division = -1
        
        % Assign negative time values before division
        for ii = index-1:-1:1
            add_list(ii) = add_list(ii+1)-1;
        end
        
        % Assign positive time values after division
        for ii = index+1:size(f_lists,1)
            add_list(ii) = add_list(ii-1)+1;
        end
        
        % Combine original data with relative time
        f_lists = [f_lists,add_list];
        
        % Separate data into mitotic and non-mitotic phases
        divide_f = [];     % Fluorescence during division window
        nondivide_f = [];  % Fluorescence outside division window
        
        for ii = 1:size(f_lists,1)
            if f_lists(ii,3)>=divide_x_min && f_lists(ii,3)<=divide_x_max
                divide_f = [divide_f;f_lists(ii,1)];
            else
                nondivide_f = [nondivide_f;f_lists(ii,1)];
            end
        end
        
        % Calculate mean fluorescence for each phase
        mean_divide = [mean_divide; mean(divide_f), mean(nondivide_f)];
        
        % Normalize data by subtracting minimum value
        y_min = min(f_lists(:,1));
        y_baseline_normalized = f_lists - y_min;
        min_max_list = [min_max_list; min(f_lists(:,1)), max(f_lists(:,1))];
        
        % Plot time series with connecting lines
        for ii = 1:size(f_lists,1)-1
            plot([f_lists(ii,3),f_lists(ii+1,3)], [y_baseline_normalized(ii,1), y_baseline_normalized(ii+1,1)], 'Color', colors(i,:), 'LineWidth', 1.5);
        end
        
        % Plot individual data points
        for ii = 1:size(f_lists,1)
            dot_size = 40;
            scatter(f_lists(ii,3), y_baseline_normalized(ii,1), dot_size, colors(1,:), 'filled');
        end
    end
end

% Format time-series plot with division window highlighting
xlim([-27,7]);
% Highlight division window in red
patch([divide_x_min,divide_x_max,divide_x_max,divide_x_min], [0,0,0.33,0.33], 'red', 'EdgeColor', 'none','FaceAlpha',0.3);
% Highlight non-division periods in blue
patch([-27,divide_x_min,divide_x_min,-27], [0,0,0.33,0.33], 'blue', 'EdgeColor', 'none','FaceAlpha',0.3);
patch([divide_x_max,7,7,divide_x_max], [0,0,0.33,0.33], 'blue', 'EdgeColor', 'none','FaceAlpha',0.3);
hold off;
%% Summary Statistics Plot - Compare mitotic vs non-mitotic fluorescence
figure;
hold on;

% Create dual y-axis plot
yyaxis left
% Plot mean fluorescence intensities for each cell lineage
for i = 1:size(mean_divide,1)
    scatter(i, mean_divide(i,1), 40, "red", 'filled', 'o');    % Mitotic phase (red circles)
    scatter(i, mean_divide(i,2), 40, "blue", 'filled', 'o');   % Non-mitotic phase (blue circles)
end
ylabel('F-intensity of mitotic and non-mitotic phases');
ax = gca;
ax.YColor = 'k';  % Set y-axis color to black
ylim([1,1.9]);    % Set fluorescence intensity range

% Right y-axis: plot relative difference between phases
yyaxis right
for i = 1:size(mean_divide,1)
    % Calculate percentage difference: (mitotic - non-mitotic) / non-mitotic
    relative_diff = (mean_divide(i,1)-mean_divide(i,2))/mean_divide(i,2);
    scatter(i, relative_diff, 40, "green", 's', 'filled');  % Green squares for difference
end
ylabel('Difference');
ax.YColor = 'k';  % Set y-axis color to black

% Format plot
grid on;
xticks(0:1:size(mean_divide,1));  % Set x-axis ticks for each cell lineage

%% Helper Function - Shrink bounding rectangle
function rect_new = rect_shrink(rect_old, shrink_ratio)
    % Shrink a bounding rectangle by the specified ratio while maintaining center
    % Input:  rect_old = [x, y, width, height], shrink_ratio (1.0 = no change)
    % Output: rect_new = [x_new, y_new, width_new, height_new]
    rect_new = [rect_old(1)+rect_old(3)*(1-shrink_ratio)/2, ...  % New x position
                rect_old(2)+rect_old(4)*(1-shrink_ratio)/2, ...  % New y position  
                shrink_ratio*rect_old(3), ...                     % New width
                shrink_ratio*rect_old(4)];                        % New height
end
