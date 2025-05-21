%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load latestfcst_ano.mat
clear all
%load gsstkaplan_raw2024_12.dat;
%rawdata = gsstkaplan_raw2024_12;
%clear gsstkaplan_raw2024_12;
%tmp=ncread('data2024_12.nc','ssta');
%tmp=ncread('data2024_12.nc','ssta');
%% 26S to 62N
lon  = ncread('ersst2024-12.nc','X');
lon = double(lon);
lat  = ncread('ersst2024-12.nc','Y');
lat = double(lat);
anom = ncread('ersst2024-12.nc','anom');
anom = double(squeeze(anom));
ssta=anom;
%%
rawdata=reshape(ssta,size(ssta,1)*size(ssta,2),size(ssta,3))';
rawdata = anomaly(rawdata);
indsea = find(rawdata(1,:));
%%
[tmp,indsst] = gsubsetindersst(rawdata,1,'NINO34');
%tdata = rmnanersst(tmp);
nino34 = mean(tmp(:,indsst),2);
ic = 317;
[pcs,eofs,sstmean1,sstmean2,sstm,teofs] = gfiltcvl3ersst(rawdata,20,ic);
%%
% figure
% plotseof(reshape(teofs(:,:,2),length(lat),length(lon)),lon,lat,'EOF-2');
% colorbar
% set(gca,'FontSize',16)
%%
nino34_5_1=ninom(pcs(1:317,1:5),eofs(:,1:5),indsst,sstmean1);
nino34_5_2=ninom(pcs(317+1:end,1:5),eofs(:,1:5),indsst,sstmean2);
nino34_5=[nino34_5_1' nino34_5_2']';

nino34_20_1=ninom(pcs(1:317,1:20),eofs(:,1:20),indsst,sstmean1);
nino34_20_2=ninom(pcs(317+1:end,1:20),eofs(:,1:20),indsst,sstmean2);
nino34_20=[nino34_20_1' nino34_20_2']';

month=1:size(pcs,1);
year = 1950+month/12;
figure
plot(year,nino34,'k')
hold on
plot(year,nino34_5,'r')
plot(year,nino34_20,'b')
legend('full',' 5 PCs', '20 PCs','Location','southeast')
xlim([year(1) year(end)])
set(gca,'FontSize',16)
title('Nino34')
