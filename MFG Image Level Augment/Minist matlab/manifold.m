
num = 100
[X,Y] = meshgrid(4:0.5:12,6:10);
Z = cos(X) + sin(Y);
CO(:,:,1) = zeros(8,19); % red
CO(:,:,2) = ones(8,19).*linspace(0.5,0.6,19); % green
CO(:,:,3) = ones(8,19).*linspace(0,1,19); % blue
s = surf(X,Y,Z,'FaceAlpha',0.5)
s.EdgeColor = 'none';
axis off
hold on
sample_x = 4+6*rand(1,num)
sample_y = 6+4*rand(1,num)
samples = cos(sample_x)+sin(sample_y)
scatter3(sample_x,sample_y,samples,'filled')
hold on
sample_x = 6+6*rand(1,num)
sample_y = 6+4*rand(1,num)
samples = cos(sample_x)+sin(sample_y)
scatter3(sample_x,sample_y,samples,'filled')