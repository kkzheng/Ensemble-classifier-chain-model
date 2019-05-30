n_protein = 690;

lg = 30;
feature_PSSM = [];
for i=1:n_protein
    dir_way = 'F:\testtesttest\Anti-inflammatory\temp_result\ppi\train\pos\';
    v_name = '.mat';
    p_way = [dir_way num2str(i-1) v_name];
    load(p_way)
    
    %data is l x 20 matrix
    y_v = 1./(1+exp(-data));
    [ FEAT ] = PseudoPSSM( y_v, lg );
    feature_PSSM=[feature_PSSM;FEAT'];
    i
end
