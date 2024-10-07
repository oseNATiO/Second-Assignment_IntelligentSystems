%% Load data and create train-test sets
hairdryer_data = readtable('hairdryer.csv');
X = table2array(hairdryer_data(:,1));
Y = table2array(hairdryer_data(:,2));

rng(4797);
[train_idx, ~, test_idx] = dividerand(size(X,1), 0.8, 0,0.2);
X_train = X(train_idx,:);
X_test = X(test_idx,:);
Y_train = Y(train_idx,:);
Y_test = Y(test_idx,:);

%% Train initial Takagi-Sugeno model
opt = genfisOptions('FCMClustering','FISType','sugeno');
opt.NumClusters = 10;
ts_model = genfis(X_train,Y_train,opt);

%% Check initial performance on test set
Y_pred_initial = evalfis(ts_model, X_test);
rmse_initial = rmse(Y_pred_initial, Y_test);
MSE_initial = rmse_initial.^2;
MAPE_initial = mape(Y_pred_initial, Y_test);
EVS_initial = explained_variance_score(Y_pred_initial, Y_test);

%Display Performance Metrics
fprintf('Initial RMSE: %4.3f \n', rmse_initial);
fprintf('Initial MSE: %4.3f \n', MSE_initial);
fprintf('Initial Mean Absolute Percentage Error: %4.3f \n', MAPE_initial);
fprintf('Initial Explained variance score: %4.3f \n', EVS_initial)

%% Tune initial model using ANFIS
[in,out,rule] = getTunableSettings(ts_model);
anfis_model = tunefis(ts_model,[in;out],X_train,Y_train,tunefisOptions("Method","anfis"));

%% Check ANFIS tuned model performance
Y_pred_final = evalfis(anfis_model, X_test);
rmse_final = rmse(Y_pred_final, Y_test);
MSE_final = rmse_final.^2; %Calculate the MSE
MAPE_final = mape(Y_pred_final, Y_test);
EVS_final = explained_variance_score(Y_pred_final, Y_test);

%Display Performance Metrics
fprintf('Final RMSE: %4.3f \n', rmse_final);
fprintf('Final MSE: %4.3f \n', MSE_final);
fprintf('Final Mean Absolute Percentage Error: %4.3f \n', MAPE_final);
fprintf('Final Explained variance score: %4.3f \n', EVS_final);

%% Function 

function evs = explained_variance_score(y_pred, y_true) 
    residual_var = var(y_true - y_pred);
    true_var = var(y_true);
    evs = 1 - (residual_var / true_var);
end