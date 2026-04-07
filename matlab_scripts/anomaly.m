function [ndata,num,season] = anomaly(data)
[length,ny,nx] = size(data);
data = data - mean(data);
ndata = zeros(length,ny,nx);
season = zeros(12,ny,nx);
num = zeros(1,12);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i = 0;
%%% this 30 years
for l=1:length
i = i+1;
   if i > 12
   i = 1;
   end 
num(i)=num(i)+1;
season(i,:,:)=season(i,:,:)+data(l,:,:);
end
for i=1:12
season(i,:,:) = season(i,:,:)/num(i);
end
i = 0;
for l=1:length
i = i+1;
   if i > 12
   i = 1;
   end 
ndata(l,:,:)=-season(i,:,:)+data(l,:,:);
end
return
