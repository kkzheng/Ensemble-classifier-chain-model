n_protein = 10;
%
%load('F:\testtesttest\datasetExtra\neg_200.mat');
%Dir_f = 'F:\testtesttest\peptide\pssm\neg\';
%val_name = '.pssm';
%fileNames_PSSM = [];
%for i=1:n_protein
%    path_way = [Dir_f num2str(i-1) val_name];
%    lujing=cellstr(path_way);
%    fileNames_PSSM = [fileNames_PSSM;lujing];
%end
%
lg = 30;
feature_PSSM = [];
for i=1:n_protein
    %
    %pp = s(i);
    %files_name = cell2mat(fileNames_PSSM(i));
    %protein_s=cell2mat(pp);
    %l_seq = size(protein_s,2);
    %PSSM_Matrix = Read_Text_files_PSSM(files_name,l_seq);
    %PSSM_Matrix is l x 20 matrix
    %
    
    dir_way = 'F:\testtesttest\Anti-inflammatory\temp_result\ppi\total\pos\';
    v_name = '.mat';
    p_way = [dir_way num2str(i-1) v_name];
    load(p_way)
    
    %data is l x 20 matrix
    y_v = 1./(1+exp(-data));
    [ FEAT ] = PseudoPSSM( y_v, lg );
    feature_PSSM=[feature_PSSM;FEAT'];
    i
end