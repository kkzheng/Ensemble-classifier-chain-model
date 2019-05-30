n_protein = 690;

feature_PSSM_DWT_P=[];
for i=1:n_protein
    dir_way = 'F:\testtesttest\Anti-inflammatory\temp_result\57pp\train\pos\';
    v_name = '.mat';
    p_way = [dir_way num2str(i-1) v_name];
    load(p_way)
    P = data';
    % GetDWT() paramter is 20 x l matrix
	FF = GetDWT(P);
    feature_PSSM_DWT_P(i,:)=FF(:);i
end
