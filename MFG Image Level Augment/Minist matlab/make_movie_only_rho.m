%x=1:N;
function make_movie_only_rho(rho, source_image_label, target_image_label)
M1 =28%space discretization [0,1.0]
M2 = 28
N=28 %time discretization
ht=1.0/(N); %time step
hx1 = 1.0/M1;
hx2= 1/M2;
domain_x = 1;
domain_y = 1;
%[x y] = meshgrid(x,x);
xxx = 0:hx1:domain_x - hx1;
yyy =0:hx2:domain_y - hx2;
[X,Y] = meshgrid(xxx,yyy);
figure(1)
file_name = append('.\Movie Results\', num2str(source_image_label),' to ', num2str(target_image_label), '.mp4');
vidfile = VideoWriter(file_name,'MPEG-4');
open(vidfile);
%obs = u(:,:,1);
for ind = 1:N
    
    imshow(rho(:,:,ind));
%     subplot(1,2,1);
%     z1=squeeze(rho(:,:,ind));
%     imagesc(xxx,yyy,z1);
%     title('\rho');
%     xlabel('x_1');
%     ylabel('x_2');
%     axis square
%     ax =gca;
%     ax.FontSize = 16;
%     subplot(1,2,2);
%     b = surf(X,Y,z1');
%     colorbar;
%     title('\rho');
%     xlabel('x_1');
%     ylabel('x_2');
%     axis square
    
    
    ax =gca;
ax.FontSize = 6;
    
    drawnow
    
    index_record =ind*3-2;
    F(index_record) = getframe(gcf);
    writeVideo(vidfile,F(index_record));
    index_record = 3*ind-1;
       F(index_record) = getframe(gcf);
    writeVideo(vidfile,F(index_record));
    index_record = 3*ind;
       F(index_record) = getframe(gcf);
    writeVideo(vidfile,F(index_record));
end
close(vidfile)
end