%% Load data and create train-test sets
clear;clc

% Load dataset and remove rows with missing values
wbco_data = readtable('wbco.csv');
wbco_data = rmmissing(wbco_data);

% Gather predictors and output
X = table2array(wbco_data(:,1:9));
Y = table2array(wbco_data(:,10));
rng(4797);

% Train and test split
train_test_partition = cvpartition(Y,'Holdout',0.2,'Stratify',true);
train_idx = training(train_test_partition);
test_idx = test(train_test_partition);
X_train = X(train_idx,:);
X_test = X(test_idx,:);
Y_train = Y(train_idx,:);
Y_test = Y(test_idx,:);

%% Train initial Takagi-Sugeno model
opt = genfisOptions('FCMClustering','FISType','sugeno');
opt.NumClusters = 5;
ts_model = genfis(X_train,Y_train,opt);

%% Check initial performance on test set

% Get model prediction and binarize output between 0 or 1 with threshold at 0.5
Y_pred_initial = evalfis(ts_model, X_test);
Y_pred_initial(Y_pred_initial>=0.5) = 1;
Y_pred_initial(Y_pred_initial<0.5) = 0;

% Get model performance metrics
initial_class_report = classperf(Y_test, Y_pred_initial);

% Retrieve and calculate relevant metrics
initial_confusion_matrix = initial_class_report.DiagnosticTable;
initial_accuracy = initial_class_report.CorrectRate;
initial_recall = initial_class_report.Sensitivity;
initial_precision = initial_class_report.PositivePredictiveValue;
initial_F1_score = 2/(1/initial_precision+1/initial_recall);
initial_Kappa_score = 0.01^2*(sum(initial_confusion_matrix(:,1))*sum(initial_confusion_matrix(1,:))+sum(initial_confusion_matrix(:,2))*sum(initial_confusion_matrix(2,:)));

% Display metrics
fprintf('Initial Accuracy: %4.3f \n', initial_accuracy);
fprintf('Initial Recall: %4.3f \n', initial_recall);
fprintf('Initial Precision: %4.3f \n', initial_precision);
fprintf('Initial F1-Score: %4.3f \n', initial_F1_score);
fprintf('Initial Kappa Score: %4.3f \n', initial_Kappa_score);

%% Tune initial model using ANFIS
[in,out,rule] = getTunableSettings(ts_model);
anfis_model = tunefis(ts_model,[in;out],X_train,Y_train,tunefisOptions("Method","anfis"));

%% Check ANFIS tuned model performance

% Get model prediction and binarize output between 0 or 1 with threshold at 0.5
Y_pred_final = evalfis(anfis_model, X_test);
Y_pred_final(Y_pred_final>=0.5) = 1;
Y_pred_final(Y_pred_final<0.5) = 0;

% Get model performance metrics
final_class_report = classperf(Y_test, Y_pred_final);

% Retrieve and calculate relevant metrics
final_confusion_matrix = final_class_report.DiagnosticTable;
final_accuracy = final_class_report.CorrectRate;
final_recall = final_class_report.Sensitivity;
final_precision = final_class_report.PositivePredictiveValue;
final_F1_score = 2/(1/final_precision+1/final_recall);
final_Kappa_score = 0.01^2*(sum(final_confusion_matrix(:,1))*sum(final_confusion_matrix(1,:))+sum(final_confusion_matrix(:,2))*sum(final_confusion_matrix(2,:)));

% Display metrics
fprintf('Final Accuracy: %4.3f \n', final_accuracy);
fprintf('Final Recall: %4.3f \n', final_recall);
fprintf('Final Precision: %4.3f \n', final_precision);
fprintf('Final F1-Score: %4.3f \n', final_F1_score);
fprintf('Final Kappa Score: %4.3f \n', final_Kappa_score);