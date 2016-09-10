%%  SVM Spam Classification with SpamAssassin corpus
%   link: http://spamassassin.apache.org/publiccorpus/
%   
%   Spam list:  20050311_spam_2.tar.bz2
%   Ham list:   20030228_easy_ham.tar.bz2
%   Total number of emails 3897 emails
%
%   By Issam, pieces of the code are from Machine Learning
%             course at Coursera (MOOC paltform).

%% Initialization
clear ; close all; clc

% ================= Loading Raw Emails ===================

% preprocess the email files by adding them to cell matrix
% Comment if your already have the cell matrix 
file_contents = emails2cellmat();

% loading the preprocessed data
% Comment if you need to load raw email to cell matrix (first time)
%file_contents = "";
%load("corpus_unextracted.mat");

% ================= Extracting Features ===================

fprintf('\nExtracting features from train emails\n');

% Extract Features
for i = 1:size(file_contents)
    fprintf('Iteration %d \n', i);
    word_indices  = processEmail(file_contents(i,:));
    X(i,:) = emailFeatures(word_indices);
end

% saving features X matrix
save corpus_X_y.mat X;

% Print Stats
fprintf('Length of feature vector:'); disp(size(X));
fprintf('\n\n');

%  % load X and y matrices if already extracted
%  % uncomment if needed
%  fprintf('\nloading features from trained emails\n');
%  load("corpus_X_y.mat");

% shuffling the examples
shuffled = [y X](randperm(size([y X],1)),:);
X = shuffled(:,2:end);
y = shuffled(:,1);

% save features and y matrices
save corpus_X_y.mat X y;

% dividing emails to train, cross and test sets
m = length(y);
% train
fprintf('\ntrain\n');
t = m * 0.6;
train_mat = shuffled(1:t, :);
Xtrain = train_mat(:, 2:end);
ytrain = train_mat(:, 1);

disp(size(Xtrain));
disp(size(ytrain));

% cross
fprintf('\ncross\n');
cross_mat = shuffled(t+1:t+m*0.2, :);
Xcross = cross_mat(:, 2:end);
ycross = cross_mat(:, 1);

disp(size(Xcross));
disp(size(ycross));

% test
fprintf('\ntest\n');
test_mat = shuffled(t+m*0.2+1:end, :);
Xtest = test_mat(:, 2:end);
ytest = test_mat(:, 1);

disp(size(Xtest));
disp(size(ytest));

% save train, cross and test matrices
save corpus_sets.mat Xtrain ytrain Xcross ycross Xtest ytest;

%% =========== Training Linear SVM ========

fprintf('\nTraining Linear SVM\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.1;
model = svmTrain(Xtrain, ytrain, C, @linearKernel);

p = svmPredict(model, Xtrain);

% save svm training model
save corpus_model.mat model

fprintf('Training Accuracy: %f\n', mean(double(p == ytrain)) * 100);

%% ========== Test Spam Classification ================

fprintf('\nExamining trained Linear SVM on the test set ...\n')

p = svmPredict(model, Xtest);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);

%% ========= Top Predictors of Spam ====================

% Sort the weights and obtin the vocabulary list
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end
