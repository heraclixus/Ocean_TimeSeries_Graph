function [subdata,indsst] = gsubsetindersst(data,extrop,box);
[nrow,ncol0] = size(data);
%%%%% these are for ip_60 dataset:27.5:292.5;-27.5:62.5%%%%%%%%%
nx = 180;
if extrop == 1 
ny = 45;
else 
ny = 13;
end
subdata = zeros(nrow,ny*nx); 
l = 0;
for j=1:ny
for i=1:nx
l=l+1;
subdata(:,l)=data(:,l);
end
end
[miss] = find(subdata(1,:)<=-998);
[l1,nmiss] = size(miss);
ncol = ny*nx; 
x = zeros(ncol,1);
y = zeros(ncol,1);
xm = zeros(ncol-nmiss,1);
ym = zeros(ncol-nmiss,1);
d = 2;
m = 0;
for j=1:ny
for i=1:nx
m=m+1;
%x(m) = 27.5 + (i-1)*d;
%y(m) = -27.5 + (j-1)*d; 
x(m) = 0 + (i-1)*d;
y(m) = -27 + (j-1)*d; 
end
end
l=0;
for i=1:ncol
if subdata(1,i) > -999
l = l+1;
xm(l) = x(i);
ym(l) = y(i);
end 
end
indsst = [];
for i=1:ncol-nmiss
% next is for NINO3 region (150W-90W);
if xm(i) >= 210 & xm(i) <= 270 & ym(i) >= -6 & ym(i) <= 6 & strcmp(box,'NINO3') 
indsst = [indsst i];
%% next is for NINO3,4 region (170W-120W);
elseif xm(i) >= 190 & xm(i) <= 240 & ym(i) >= -6 & ym(i) <= 6 & strcmp(box,'NINO34') 
indsst = [indsst i];
%% this is for CEI
elseif xm(i) >= 50 & xm(i) <= 80 & ym(i) >= -15 & ym(i) <= 0 & strcmp(box,'CEI') 
indsst = [indsst i];
% this is for   TNA
elseif xm(i) >= 320 & xm(i) <= 340 & ym(i) >= 5 & ym(i) <= 20 & strcmp(box,'TNA') 
indsst = [indsst i];
% this is for   TSA
elseif xm(i) >= 345 & xm(i) <= 360 & ym(i) >= -15 & ym(i) <= -5& strcmp(box,'TSA') 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
indsst = [indsst i];
else
end
end

return
