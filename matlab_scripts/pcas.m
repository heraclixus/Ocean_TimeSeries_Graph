function [pcs,eofs,d] = pcas(data,npc);
%%%% computes only npc of eofs, not all
[length ncol] = size(data);
covmat = data'*data/length;
%covmat = cov(data);	
opts.disp = 0;
[eofs,d] = eigs(covmat,npc,'LM',opts);
pcs = data*eofs;
d=diag(d);
return
