clear; close all;

% Design of Experiments (DoE)

% Outputs are 'all_experiments.csv' and 'selected_experiments.csv'
% 'all_experiments.csv' includes all possible experiments and
% 'selected_experiments.csv' includes experiments selected from them.

% In settings, you can change the contents of 'variable1', 'variable2' and 'variable3',
% delete 'variable3', and add 'variable4', 'variable5', ... after 'variable3'
    
number_of_experiments = 30;
variables = cell(1);
variables{1} = [1,2,3,4,5];
variables{2} = [-10, 0, 10, 20];
variables{3} = [0.2, 0.6, 0.8, 1, 1.2];

%% make all possible experiments
all_experiments = variables{1}'; 
for variablenum = 2 : size( variables, 2)
    grid_seed = variables{variablenum};
    grid_tmp = repmat( grid_seed, size( all_experiments, 1 ), 1 );
    all_experiments = [ repmat( all_experiments, size(grid_seed,2), 1 ) grid_tmp(:) ];        
end
csvwrite( 'all_experiments.csv', all_experiments);

%% select experiments
autoscaled_all_experiments = zscore(all_experiments);
% selected_experiments = zeros( number_of_experiments, size( all_experiments, 2) );

for experiment_number = 1 : size( all_experiments, 1) - number_of_experiments
    determinants = zeros( size( all_experiments, 1), 1);
    autoscaled_all_experiments_tmp = autoscaled_all_experiments;
    for calc_determinant_number = 1 : size( all_experiments, 1)
        autoscaled_all_experiments( calc_determinant_number, :) = [];
%             determinants( calc_determinant_number) = sum(sum( autoscaled_all_experiments' * autoscaled_all_experiments));
        determinants( calc_determinant_number) = det( autoscaled_all_experiments' * autoscaled_all_experiments);
        autoscaled_all_experiments = autoscaled_all_experiments_tmp;
    end    
    selected_number = find( determinants == max( determinants) );

%     selected_experiments( experiment_number, :) = all_experiments( selected_number(1), :);
    all_experiments( selected_number(1), :) = [];
    autoscaled_all_experiments( selected_number(1), :) = [];
end

% for experiment_number = 1 : number_of_experiments
%     disp( [ num2str(experiment_number) ' / ' num2str(number_of_experiments) ]);
%     if experiment_number == 1
%         selected_number = 1;
%     else
%         determinants = zeros( size( all_experiments, 1), 1);
% %         selected_experiments_tmp = selected_experiments;
%         for calc_determinant_number = 1 : size( all_experiments, 1)
%             selected_experiments_tmp = [selected_experiments; autoscaled_all_experiments( calc_determinant_number, :)];
% %             autoscaled_all_experiments( calc_determinant_number, :) = [];
% %             determinants( calc_determinant_number) = sum(sum( autoscaled_all_experiments' * autoscaled_all_experiments));
%             determinants( calc_determinant_number) = det( autoscaled_all_experiments' * autoscaled_all_experiments);
% %             autoscaled_all_experiments = autoscaled_all_experiments_tmp;
%         end    
%         selected_number = find( determinants == max( determinants) );
% %         selected_number
%     end
% 
%     selected_experiments( experiment_number, :) = all_experiments( selected_number(1), :);
%     all_experiments( selected_number(1), :) = [];
%     autoscaled_all_experiments( selected_number(1), :) = [];
% end

csvwrite( 'selected_experiments.csv', all_experiments);
