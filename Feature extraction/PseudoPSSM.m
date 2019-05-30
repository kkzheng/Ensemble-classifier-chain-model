function [ FEAT ] = PseudoPSSM( P, lg )
%PR: protein sequence
%P:PSSM matrix of the protein PR
%lg=max value of lag

n=size(P,1);%length of the protein

%for i=1:n
%    ME(i)=mean(P(i,:));
%    SD(i)=std(P(i,:));
%    V(i,:)=(P(i,:)-ME(i))./SD(i);
%end
V=P;
V(find(isinf(V)))=0;
V(find(isnan(V)))=0;

AC=zeros(20,lg);
for lag=1:lg
    for i=1:20
        for j=1:n-lag
            AC(i,lag)=AC(i,lag)+(V(j,i)-V(j+lag,i))^2;
        end
        AC(i,lag)=AC(i,lag)./(n-lag);
    end
end
AC(find(isinf(AC)))=0;
AC(find(isnan(AC)))=0;
%FEAT=[single(AC(:)); mean(V)']; % AC 20*30  V L*20  mean 换成一行1*20  single(AC(:)) 用一列表示?600*1  620行
FEAT=[mean(V)';single(AC(:))];
FEAT=single(FEAT(:));