
% 
% for nr = 10:30
% path = ("\\AD.utwente.nl\TNW\BMPI\Archive\PEOPLE ALL\PROJECTS MASTER\2023 - Eline Zoetelief\Data\Badmat_verwerkte data\Group 3\Number " +nr);
% 
% 
% load(path + "\Masked_images_dorsal.mat")
% Dorsal_Left_crop = img_mask_left;
% Dorsal_Right_crop =img_mask_right;
% 
% load(path + "\Dorsal_images.mat")
% Dorsal_Left = feet_left;
% Dorsal_Right = feet_right;
% 
% load(path + "\DFU_images.mat")
% Direct_plantar_Left = feet_left{1,1};
% Direct_plantar_Right = feet_right{1,1};
% Indirect_plantar_Left = feet_left{2,1};
% Indirect_plantar_Right = feet_right{2,1};
% 
% load(path + "\Cropped_images.mat")
% Direct_plantar_Left_crop = left_feet_image{1,1};
% Direct_plantar_Right_crop = right_feet_image{1,1};
% Indirect_plantar_Left_crop = left_feet_image{2,1};
% Indirect_plantar_Right_crop = right_feet_image{2,1};
% 
% 
% save (("pnt"+string(31-nr)), "Indirect_plantar_Right_crop", "Indirect_plantar_Left_crop", "Direct_plantar_Right_crop", "Direct_plantar_Left_crop", "Indirect_plantar_Right", "Indirect_plantar_Left", "Direct_plantar_Right", "Direct_plantar_Left", "Dorsal_Right", "Dorsal_Left", "Dorsal_Right_crop", "Dorsal_Left_crop")
% 
% 
% end

figure(1), 
imagesc(Indirect_plantar_Left_crop)
colormap('hot')
clim([20 40]);
set(gca, 'XTickLabel', [])
set(gca, 'YTickLabel', [])
