xxx = hx1:hx1:1;
yyy = hx2:hx2:1;
[X,Y] = meshgrid(xxx,yyy);
figure(111)
c = zeros(N,1);
%obs = u(:,:,1);
for ind = 1:N
    
    subplot(8,8,ind);
    z1=squeeze(rho(:,:,ind));
    %imagesc(z)%,colormap(hot)
    c(ind) = sum(sum(z1));
    b = imagesc(z1);%surf(X,Y,z1);
    %alpha 0.2
   % hold on
    %b.EdgeColor = 'none';
%    s=surf(X,Y,obs);
%     s.FaceAlpha=0.5;
%     s.EdgeColor = 'cyan';
   % s.FaceColor = 'g';
   % axis([0 1 0 1 0 65])
end
figure(101)
plot(c);

figure(102);
for ind = 1:N
    
    subplot(8,8,ind);
    z1=squeeze(phi(:,:,ind));
    %imagesc(z)%,colormap(hot)
    
    b = surf(X,Y,z1);
    %alpha 0.2
    hold on
    %b.EdgeColor = 'none';
   % s=surf(X,Y,obs);
   % s.FaceAlpha=0.5;
   % s.EdgeColor = 'cyan';
   % s.FaceColor = 'g';
   % axis([0 1 0 1 0 65])
end