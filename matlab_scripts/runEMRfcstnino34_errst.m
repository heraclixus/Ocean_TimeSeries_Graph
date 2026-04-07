%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
%pause %%%%%%%%%%%%%%%%%%%%%%%%%
lon  = ncread('ersst2024-12.nc','X');
lon = double(lon);
lat  = ncread('ersst2024-12.nc','Y');
lat = double(lat);
anom = ncread('ersst2024-12.nc','anom');
anom = double(squeeze(anom));
ssta=anom;
rawdata=reshape(ssta,size(ssta,1)*size(ssta,2),size(ssta,3))';
rawdata = anomaly(rawdata);
[tmp,indsst] = gsubsetindersst(rawdata,1,'NINO34');
tdata = rmnanersst(tmp);
nino34 = mean(tdata(:,indsst),2);
ic = 317;
[pcs,eofs,sstmean1,sstmean2,sstm,teofs] = gfiltcvl3ersst(rawdata,20,ic);


save("rawdata.mat", "rawdata");
save("tdata.mat", "tdata");
save("nino34.mat", "nino34");
save("indsst.mat", "indsst");
save("pcs.mat", "pcs");


%%
MM=20;
data=pcs(:,1:MM);
stddata=std(data);
NT=size(pcs,1);
numTimeStepsTrain = 700;
ipls = 0;
ires=0;
inorm = 0;
lead = 12;  %% maximum prediction lead time

N1 = numTimeStepsTrain; %% end of the model training interval as in [1 N1];
N2 = numTimeStepsTrain+1;  %% start of the cross-validation interval as in [N2 NE];
N2=2;
NE = NT-lead; %% end of the cross-validation interval  as in [N2 NE];

%period=[];%%
period=12;


nelin = 1;
nlevel = 2;
iext=1;
cnoise=0;
niter=1;
[fcst,verf,rms,anc,modstr,xt_res]= fcstemrplsiext2(data,period,nelin,nlevel,niter,inorm,ipls,lead,N1,N2,NE,iext,cnoise);
%%
fcstm=mean(fcst,4);
sstn = zeros(size(fcstm,1),lead);
sstv = zeros(size(fcstm,1),lead);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for l=1:lead
sstv(:,l)=ninom(squeeze(verf(:,l,:)),eofs,indsst,sstmean2);
sstn(:,l)=ninom(squeeze(fcstm(:,l,:)),eofs,indsst,sstmean2);
end
%%
     
         NTOT=size(sstn,1);
         lead=size(sstn,2);
         figure
         for i=1:8
         LL=i;
         subplot(4,2,i)
         indt=N2+LL:N2+LL+NTOT-1;
         plot(indt,sstv(:,LL),'b')
         hold on
         plot(indt,sstn(:,LL),'r')
         xlim([indt(1) indt(end)])
         title(num2str(LL));
         legend('data','fcst','Location','northwest')
         set(gca,'FontSize',16')
         end
        
         
     
     
     anc = zeros(lead,1);
     rms = zeros(lead,1);
     stdt= squeeze(std(sstv));
     ldat=size(sstv,1);
  
    
     for i=1:lead
     rms(i)=sqrt(sum((squeeze(sstn(:,i))-squeeze(sstv(:,i))).^2)/ldat)/stdt(i);
     anc(i)=xcorr(center(squeeze(sstn(:,i))),center(squeeze(sstv(:,i))),0,'coeff');
     end
     
        figure
        plot(rms,'r')
        hold on
        plot(anc,'b')
        xlim([1 lead])
        ylim([0 1])
        legend('RMSE','Corr')
        grid on
        set(gca,'FontSize',16)
        xlabel('lead')
     

  










