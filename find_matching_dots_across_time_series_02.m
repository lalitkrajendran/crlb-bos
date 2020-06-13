function [X_matched, Y_matched, Z_matched, d_x_matched, d_y_matched, R_matched, I_matched] = find_matching_dots_across_time_series_02(X, Y, Z, d_x, d_y, R, I)
    % ---------------------------------------------------------------------
    % find matching dots between successive frames in the reference images
    % ---------------------------------------------------------------------
    s_radius = 1;
    weights = [1, 1, 1];
    fprintf('finding nearby dot locations in the reference image series\n');
    num_images = numel(X);
    % cell array to hold all tracks
    tracks = cell(1, num_images);
    % perform tracking across all frames
    parfor image_index = 1:num_images
        fprintf('image: %d\n', image_index);
        [tracks{image_index}]=weighted_nearest_neighbor3D(X{1},X{image_index},X{1},Y{1},Y{image_index},Y{1},...
                    Z{1},Z{image_index},Z{1},d_x{1},d_x{image_index},I{1},I{image_index},weights,s_radius);

    end

    fprintf('extracting position of identified dots in reference images across successive frames\n');

    % arrays to hold co-ordinates of matched dots. each row corresponds to
    % positions of the same dot across the series of grad images.
    X_matched = ones(size(X{1},1), num_images) * NaN;
    Y_matched = ones(size(Y{1},1), num_images) * NaN;
    Z_matched = zeros(size(Z{1},1), num_images);
    d_x_matched = ones(size(d_x{1},1), num_images) * NaN;
    d_y_matched = ones(size(d_x{1},1), num_images) * NaN;
    R_matched = ones(size(I{1},1), num_images) * NaN;
    I_matched = ones(size(I{1},1), num_images) * NaN;

    % find matching dots across the time series
    for p = 1:size(X{1},1)
        parfor image_index = 1:num_images
            % find row with same particle index
            track_index = find(tracks{image_index}(:,11) == p);
            % find id of matched dot
            matched_dot_index = tracks{image_index}(track_index, 12);
            % if a row was found, then copy the co-ordinates
            if ~isempty(track_index)            
                X_matched(p, image_index) = tracks{image_index}(track_index, 2);
                Y_matched(p, image_index) = tracks{image_index}(track_index, 4);
                d_x_matched(p, image_index) = d_x{image_index}(matched_dot_index);
                d_y_matched(p, image_index) = d_y{image_index}(matched_dot_index);
                R_matched(p, image_index) = R{image_index}(matched_dot_index);
                I_matched(p, image_index) = I{image_index}(matched_dot_index);
            end
        end
    end
end