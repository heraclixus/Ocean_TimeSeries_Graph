function [pcs,eofs,sstmean1,sstmean2,sst,teofs] = gfiltcvl3ersst(data,npc,st1);
[miss] = find(data(1,:)<-998);
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
if st1 > 0
sstmean1 = mean(sst(1:st1,:));
sstmean2 = mean(sst(st1+1:nrow,:));
sstm1 = detrend(sst(1:st1,:),'constant');
sstm2 = detrend(sst(st1+1:nrow,:),'constant');
sstm = [sstm1' sstm2']';
else
sstmean1 = mean(sst);
sstmean2 = sstmean1;
sst0 = detrend(sst);
sst0=sst;
sstm = detrend(sst0,'constant');
end
[pcs eofs] = pcas(sstm,npc);
nx = 180;
ny = 45;
teofs = NaN*ones(ny,nx,npc);
index = zeros(ny*nx,1);

l = 0;
m = 0;
for j=1:ny
for i=1:nx
l = l+1;
 if data(1,l) > -998 
index(l)=1;
m = m+1;
teofs(j,i,:) = eofs(m,:);
 end
end
end


return
