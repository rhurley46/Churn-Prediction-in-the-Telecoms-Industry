%% 1. Load Data
% The first step is to load the data into matlab

data = readtable('churn_pp.csv');

%% 2. Balance the Data
% The data is unbalanced by a ration of ~5:2. This will be rebalanced using
% ADASYN

%convert to array for adasyn
data = table2array(data);

%create new synthetic data
adasyn_data= ADASYN(data(:,1:42), data(:,43),1,5,5,false);

%populate empty adasyn column with target variables ('1')
adasyn_data(:, 43) = 1;

%combine real data with synthetic adasyn data
concat = vertcat(data, adasyn_data);

%round to 0 or 1
rounddata = round(concat);

%% 3. Save to .mat
% The new dataset is saved to a .mat file for reproducability

% save('BalancedData.mat','rounddata');

%% 4. Load the new file as an array

balanceddata = load('BalancedData.mat');
data =  balanceddata.rounddata;

%% 5. Preprocessing
%splits data into training and test sets

rng('default');
% Cross varidation (train: 80%, test: 20%)
cv = cvpartition(size(data,1),'HoldOut',0.2);
idx = cv.test;
% Separate to training and test data
Training = data(~idx,:);
Testing = data(idx,:);

%% 6. Save Splits to CSV files

save('Training.mat','Training');
save('Testing.mat','Testing');
