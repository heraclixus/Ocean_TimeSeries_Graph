function [sst,miss] = rmnan(data);
[miss] = find(data(1,:)<=-998);
[l1,nmiss] = size(miss);
[nrow, ncol] = size(data);
sst = zeros(nrow,ncol-nmiss);
l=0;
for i=1:ncol
if data(1,i) > -998
l = l+1;
sst(:,l) = data(:,i);
end
end
return
