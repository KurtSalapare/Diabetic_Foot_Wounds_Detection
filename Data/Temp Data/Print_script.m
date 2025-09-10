for i = 1:30
load(strcat("pnt", num2str(i), ".mat"))

fh = figure('Menu','none','ToolBar','none'); 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
imagesc(Indirect_plantar_Right)
colormap('hot')
axis off
print(strcat("pnt", num2str(i), "IND_R"),'-dpng')
close all

fh = figure('Menu','none','ToolBar','none'); 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
imagesc(Indirect_plantar_Left)
colormap('hot')
axis off
print(strcat("pnt", num2str(i), "IND_L"),'-dpng')
close all

fh = figure('Menu','none','ToolBar','none'); 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
imagesc(Dorsal_Right)
colormap('hot')
axis off
print(strcat("pnt", num2str(i), "DOR_R"),'-dpng')
close all

fh = figure('Menu','none','ToolBar','none'); 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
imagesc(Dorsal_Left)
colormap('hot')
axis off
print(strcat("pnt", num2str(i), "DOR_L"),'-dpng')
close all

end