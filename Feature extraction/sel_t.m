fea = [];

dir_way = 'F:\testtesttest\Anti-inflammatory_repeate\final_result\total\pos3.mat';
load(dir_way)
fea = [fea;feature];
dir_way = 'F:\testtesttest\Anti-inflammatory_repeate\final_result\total\neg3.mat';
load(dir_way)
fea = [fea;feature];

sz = size(fea);
feaNum = sz(2);
sampleNum = sz(1);
features = [];

for i = 1 : feaNum
    [h,p,ci,stats] = ttest2(fea(1:863,i),fea(864:2124,i));
    if p <= 0.005
        features = [features,fea(:,i)];
    end
end

size(features)

pos = features(1:863,:);
train_pos = pos(1:690,:);
val_pos = pos(691:863,:);

neg = features(864:2124,:);
train_neg = neg(1:1009,:);
val_neg = neg(1010:1261,:);

feature_train = [train_pos;train_neg];
feature_val = [val_pos;val_neg];