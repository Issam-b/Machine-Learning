function file_contents = emails2cellmat()
%EMAILS2CELLMAT load email files and return the result as 
%cell matrix
%   file_contents = EMAILS2CELLMAT() load email files and 
%   return the result as cell matrix

%% ==================== Spam Preprocessing ====================

fprintf('\nLoading spam files\n');

% list directory spam files
spam_files = readdir("spam_2/");
spam_files = spam_files(3:end,:);

% setting y to 1 (spam)
y_spam = ones(length(spam_files), 1);
file_contents_spam = "";

% read each spam file to file_contents_spam matrix
for i = 1:size(spam_files, 1);
    filename = strcat("spam_2/", spam_files(i)){1};
    file_contents_spam = [ file_contents_spam; readFile( filename )];
end

disp(size(file_contents_spam));

%% =================== Ham Preprocessing =====================

fprintf('\nLoading ham files\n');

% list directory spam files
ham_files = readdir("easy_ham/");
ham_files = ham_files(3:end,:);

% setting y to 1 (spam)
y_ham = zeros(length(ham_files), 1);
file_contents_ham = "";

% read each spam file to file_contents_ham matrix
for i = 1:size(ham_files, 1);
    filename = strcat("easy_ham/", ham_files(i)){1};
    file_contents_ham = [ file_contents_ham; readFile( filename )];
end

fprintf('\nspam size:'); disp(size(file_contents_spam));
fprintf('ham size:'); disp(size(file_contents_ham));

% merge spam and ham emails
file_contents = [file_contents_spam; file_contents_ham];
y = [y_spam; y_ham];
fprintf('\nX (all) size:'); disp(size(file_contents));
fprintf('y (all) size:'); disp(size(y));

% save matrices before extracting features
save corpus_unextracted.mat file_contents y;

end
