f0 = load('plot_T0_A0_A0.dat');
f1 = load('plot_T1_A0_A0.dat');
f0 = reshape(f0, 41, 41, 41);
f1 = reshape(f1, 41, 41, 41);
x = -2:0.1:2;

close all
figure(1)
patch(isocaps(f0,.95),...
   'FaceColor','blue','EdgeColor','blue');
p1 = patch(isosurface(f0,.95),...
   'FaceColor','blue','EdgeColor','blue');
isonormals(f0,p1)
view(3); 
axis vis3d tight
camlight left; 
%colormap jet
lighting gouraud

hold on

patch(isocaps(-f0,-.05),...
   'FaceColor','red','EdgeColor','red');
p2 = patch(isosurface(-f0,-.05),...
   'FaceColor','red','EdgeColor','red');
isonormals(-f0,p2)
view(3); 
axis vis3d tight
camlight left; 
%colormap jet
lighting gouraud

set(gcf,'units','inches');
set(gcf,'position',[0 0 10 8])
set(gcf, 'PaperUnits', 'inches', 'PaperPosition',[0 0 10 8])

figure(2)
patch(isocaps(f1,.95),...
   'FaceColor','blue','EdgeColor','blue');
p1 = patch(isosurface(f1,.95),...
   'FaceColor','blue','EdgeColor','blue');
isonormals(f1,p1)
view(3); 
axis vis3d tight
camlight left; 
%colormap jet
lighting gouraud

hold on

patch(isocaps(-f1,-.05),...
   'FaceColor','red','EdgeColor','red');
p2 = patch(isosurface(-f1,-.05),...
   'FaceColor','red','EdgeColor','red');
isonormals(-f1,p2)
view(3); 
axis vis3d tight
camlight left; 
%colormap jet
lighting gouraud

set(gcf,'units','inches');
set(gcf,'position',[0 10 10 8])
set(gcf, 'PaperUnits', 'inches', 'PaperPosition',[0 0 10 8])

% for i = 1:41
%     figure(1)
%     contourf(x,x,f0(:,:,i))
%     caxis([0,1]);
%     colorbar
%     set(gca,'FontSize',15, 'FontName', 'Arial', 'LineWidth',2,'GridLineStyle','-','MinorGridLineStyle','-')
%     xlabel(x(i))
%     set(gcf,'units','inches');
%     set(gcf,'position',[0 0 10 8])
%     set(gcf, 'PaperUnits', 'inches', 'PaperPosition',[0 0 10 8])
%     
%     figure(2)
%     contourf(x,x,f1(:,:,i))
%     caxis([0,1]);
%     colorbar
%     set(gca,'FontSize',15, 'FontName', 'Arial', 'LineWidth',2,'GridLineStyle','-','MinorGridLineStyle','-')
%     xlabel(x(i))
%     set(gcf,'units','inches');
%     set(gcf,'position',[0 10 10 8])
%     set(gcf, 'PaperUnits', 'inches', 'PaperPosition',[10 0 10 8])
%     pause(1)
% end

