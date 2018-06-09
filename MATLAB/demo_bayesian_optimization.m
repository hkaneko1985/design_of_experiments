clear; close all;
% Demonstration of Bayesian optimization

%% settings
number_of_samples = 10000;
number_of_first_samples = 30;
number_of_iteration = 100;
acquisition_function_flag = 2; % 1: Mutual information (MI), 2: Expected Improvement(EI), 3: Probability of improvement (PI) [0: Estimated y-values]
do_maximization = false; % true: maximization, false: minimization

%% generate dataset
rng('default');
rng(4);
X = rand(number_of_samples,2)*4-2;
function1 = 1 + ((X(:,1) + X(:,2) + 1).^2) .* (19 - 14*X(:,1) + 3*X(:,1).^2 - 14*X(:,2) + 6*X(:,1).*X(:,2) + 3*X(:,2).^2);
function2 = 30 + ((2*X(:,1) - 3*X(:,2)).^2) .* (18 - 32*X(:,1) + 12*X(:,1).^2 + 48*X(:,2) - 36*X(:,1).*X(:,2) + 27*X(:,2).^2);
y = log(function1 .* function2);
disp( ['min of y : ' num2str(min(y)) ]);
original_X = X;
original_y = y;

figure;
scatter3(X(:,1),X(:,2),y,[],y,'filled');
colormap(jet);
colorbar;
xlim([ min(original_X(:,1))-range(original_X(:,1))*0.03 max(original_X(:,1))+range(original_X(:,1))*0.03] );
ylim([ min(original_X(:,2))-range(original_X(:,2))*0.03 max(original_X(:,2))+range(original_X(:,2))*0.03] );
zlim([ min(original_y)-range(original_y)*0.03 max(original_y)+range(original_y)*0.03] );
axis square;
xlabel( 'x_1' ,  'FontSize' , 18 , 'FontName', 'Times');
ylabel( 'x_2' ,  'FontSize' , 18 , 'FontName', 'Times');
zlabel( 'y' ,  'FontSize' , 18 , 'FontName', 'Times');
set(gcf, 'Color' , 'w' ); 
set(gca, 'FontSize', 18);
set(gca, 'FontName', 'Times');

%% set first samples
bad_sample_number = find( y > 10 );
% bad_sample_number = 1:length(y);
high_X = X(bad_sample_number, :);
high_y = y(bad_sample_number);
random_numbers = randperm(length(high_y));
rng('shuffle');
first_sample_numbers = random_numbers(1:number_of_first_samples);
X_train = high_X( first_sample_numbers, :);
y_train = high_y( first_sample_numbers, :);
X(bad_sample_number(first_sample_numbers), :) = [];
y(bad_sample_number(first_sample_numbers), :) = [];

figure;
scatter3(X_train(:,1),X_train(:,2),y_train,[],'b','filled');
xlim([ min(original_X(:,1))-range(original_X(:,1))*0.03 max(original_X(:,1))+range(original_X(:,1))*0.03] );
ylim([ min(original_X(:,2))-range(original_X(:,2))*0.03 max(original_X(:,2))+range(original_X(:,2))*0.03] );
zlim([ min(original_y)-range(original_y)*0.03 max(original_y)+range(original_y)*0.03] );
axis square;
xlabel( 'x_1' ,  'FontSize' , 18 , 'FontName', 'Times');
ylabel( 'x_2' ,  'FontSize' , 18 , 'FontName', 'Times');
zlabel( 'y' ,  'FontSize' , 18 , 'FontName', 'Times');
set(gcf, 'Color' , 'w' ); 
set(gca, 'FontSize', 18);
set(gca, 'FontName', 'Times');

%% Bayesian optimization
if ~do_maximization
    y = -y;
    y_train = -y_train;
end
cumulative_variance = zeros(length(y), 1);
[selected_candidate_number, selected_X_candidate, cumulative_variance] = bayesian_optimization(X_train, y_train, X, acquisition_function_flag, cumulative_variance );
disp(['next experiment : ' num2str(selected_X_candidate)]);
