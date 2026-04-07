%%
         load skill.mat
         NTOT=size(sstn,1);
         lead=size(sstn,2);
         figure
         for i=1:8
         LL=i*3;
         subplot(4,2,i)
         indt=N2+LL:N2+LL+NTOT-1;
         plot(indt,sstv(:,LL),'b')
         hold on
         plot(indt,sstn(:,LL),'r')
         xlim([indt(1) indt(end)])
         title(num2str(LL));
         legend('data','fcst','Location','southwest')
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
     
