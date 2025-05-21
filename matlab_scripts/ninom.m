function [sstn] = ninom(pcs,eofs,indsst,sstmean);    
    [length nmax] = size(pcs);
    [l1 nsst] = size(indsst);
    sstn = zeros(length,1);
    for l=1:length
      for k=1:nsst
	sstn(l) = sstn(l)+sstmean(indsst(k));
     for j=1:nmax
    sstn(l)=sstn(l)+pcs(l,j)*eofs(indsst(k),j);
      end
     end
    end
    sstn = sstn/nsst;
    return
