function [selected_candidate_number, selected_X_candidate, cumulative_variance] = bayesian_optimization(X, y, candidates_of_X, acquisition_function_flag, cumulative_variance )
% Bayesian optimization
%   Hiromasa Kaneko
%
% --- input ---
% X : m x n matrix of X-variables of training dataset (m is the number of samples and n is the number of X-variables)
% y : m x 1 vector of a y-variable of training dataset
% candidates_of_X : k x n matrix of X-variables of new experiment candidates
% acquisition_function_flag : 1: Mutual information (MI), 2: Expected improvement(EI), 3: Probability of improvement (PI) [0: Estimated y-values]
% 
% --- output ---
% selected_candidate_number : selected number of candidates_of_X
% selected_X_candidate : 1 x n vector of selected X candidate
% cumulative_variance : cumulative variance in mutual information (MI)[acquisition_function_flag=1]

if ~exist('cumulative_variance','var'), cumulative_variance = zeros(length(y), 1); end

relaxation_value = 0.01;
delta = 10^-6;
alpha = log(2/delta);


[autoscaled_X, mean_of_X_train, std_of_X_train] = zscore(X);
autoscaled_y = zscore(y);
% gaussian_process_model = CompactRegressionGP;
% gaussian_process_model.fitrgp( autoscaled_X, autoscaled_y);
% [autoscaled_estimated_y_test, autoscaled_std_of_estimated_y_test] = gaussian_process_model.predict((candidates_of_X - repmat(mean_of_X_train, size(candidates_of_X, 1), 1) ./ repmat(std_of_X_train, size(candidates_of_X,1), 1)));
gaussian_process_model = fitrgp( autoscaled_X, autoscaled_y);
[autoscaled_estimated_y_test, autoscaled_std_of_estimated_y_test] = predict(gaussian_process_model, (candidates_of_X - repmat(mean_of_X_train, size(candidates_of_X, 1), 1) ./ repmat(std_of_X_train, size(candidates_of_X,1), 1)));
switch acquisition_function_flag
    case 1
        acquisition_function_values = autoscaled_estimated_y_test + alpha ^ 0.5 * ((autoscaled_std_of_estimated_y_test.^2 + cumulative_variance) .^ 0.5 - cumulative_variance .^ 0.5);
        cumulative_variance = cumulative_variance + autoscaled_std_of_estimated_y_test .^ 2;
    case 2
        acquisition_function_values = (autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) .* ...
            normcdf( (autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) ./ autoscaled_std_of_estimated_y_test ) + ...
            autoscaled_std_of_estimated_y_test .* normpdf( (autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) ./ autoscaled_std_of_estimated_y_test );
    case 3
        acquisition_function_values = normcdf( (autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) ./ autoscaled_std_of_estimated_y_test );
    case 0
        acquisition_function_values = autoscaled_estimated_y_test;
end
selected_candidate_number = find( acquisition_function_values == max(acquisition_function_values) ); selected_candidate_number = selected_candidate_number(1);
selected_X_candidate = candidates_of_X(selected_candidate_number, :);

end

