
for i=1:30
if i<10
    mdir = strcat('pnt 0', string(i));
else
    mdir = strcat('pnt '," " , string(i));
end

mkdir(mdir);
mkdir(strcat (mdir,'/Dorsal'));
mkdir(strcat(mdir,'/Plantar'));

end
% 
% for i=1:16
% if i<10
%     mdir = strcat('gz 0', string(i));
% else
%     mdir = strcat('gz '," " , string(i));
% end
% 
% mkdir(mdir);
% 
% for j = 1:10
% mkdir(strcat (mdir,'/Dorsal/day'," ",string(j)));
% mkdir(strcat(mdir,'/Plantar/day'," ",string(j)));
% end
% end