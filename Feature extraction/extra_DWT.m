% n_protein = 4802;
% load('D:\MATLAB 2017a\wei\locations_37.mat');
% Dir_f = 'D:\MATLAB 2017a\wei\PSSM\';  
n_protein = 7;

%
%load('F:\testtesttest\datasetExtra\neg_200.mat');
%Dir_f = 'F:\testtesttest\peptide\pssm\neg\'; 
%val_name = '.pssm';
%fileNames_PSSM = [];
%for i=1:n_protein
%    path_way = [Dir_f num2str(i-1) val_name];
%    lujing=cellstr(path_way);
%	fileNames_PSSM = [fileNames_PSSM;lujing];
%end
%
feature_PSSM_DWT_P=[];
for i=1:n_protein
    %
    %Seq_Length = length(s{i});
	%files_name = cell2mat(fileNames_PSSM(i));
	%PSSM_Matrix = Read_Text_files_PSSM(files_name,Seq_Length);
	%P = PSSM_Matrix;
	%
    dir_way = 'F:\testtesttest\Anti-inflammatory\temp_result\57pp\total\pos\';
    v_name = '.mat';
    p_way = [dir_way num2str(i-1) v_name];
    load(p_way)
    P = data';
    % GetDWT() paramter is 20 x l matrix
	FF = GetDWT(P);
    feature_PSSM_DWT_P(i,:)=FF(:);i
end
