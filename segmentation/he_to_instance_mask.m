images_train_folder = '../datasets/MoNuSeg/train/Tissue Images';
annotations_train_folder = '../datasets/MoNuSeg/train/Annotations';
images_val_folder = '../datasets/MoNuSeg/val/Tissue Images';
annotations_val_folder = '../datasets/MoNuSeg/val/Annotations';

images_train_path = dir(images_train_folder);
annotations_train_path = dir(annotations_train_folder);
images_val_path = dir(images_val_folder);
annotations_val_path = dir(annotations_val_folder);
images_path = [images_train_path;images_val_path];
annotations_path = [annotations_train_path;annotations_val_path];

for i = 3:length(images_path)
    filename = split(images_path(i).name, ".");
    filename_annotation = filename{1}+".xml";
    disp(images_path(i).name)
    disp(filename)
    matching_annotation = find(ismember({annotations_path.name}, filename_annotation));
    if isempty(matching_annotation)
        disp("No matching annotation found, skipping image: ", images_path(i).name);
    else
        image_path = fullfile(images_path(i).folder,images_path(i).name);
        annotation_path = fullfile(annotations_path(matching_annotation).folder,annotations_path(matching_annotation).name);
        [binary_mask,color_mask, instance_mask] = create_instance_mask(image_path, annotation_path);
    end
    path_binary_mask = fullfile(annotations_path(i).folder,strcat(filename{1},'_binary.png'));
    path_instance_mask = fullfile(annotations_path(i).folder,strcat(filename{1},'_instance.png'));
    imwrite(binary_mask, path_binary_mask);
    imwrite(instance_mask, path_instance_mask);
end


% adaption of the he_to_binary_mask_final.m code from https://monuseg.grand-challenge.org/Data/
function [binary_mask,color_mask, instance_mask]=create_instance_mask(path_image, path_annotation)
    xml_file=path_annotation; 

    xDoc = xmlread(xml_file);
    Regions=xDoc.getElementsByTagName('Region'); % get a list of all the region tags
    for regioni = 0:Regions.getLength-1
        Region=Regions.item(regioni);  % for each region tag

        %get a list of all the vertexes (which are in order)
        verticies=Region.getElementsByTagName('Vertex');
        xy{regioni+1}=zeros(verticies.getLength-1,2); %allocate space for them
        for vertexi = 0:verticies.getLength-1 %iterate through all verticies
            %get the x value of that vertex
            x=str2double(verticies.item(vertexi).getAttribute('X'));

            %get the y value of that vertex
            y=str2double(verticies.item(vertexi).getAttribute('Y'));
            xy{regioni+1}(vertexi+1,:)=[x,y]; % finally save them into the array
        end

    end
    im_info=imfinfo(path_image);
 
    nrow=im_info.Height; %(s2)
    ncol=im_info.Width; %(s2)
    binary_mask=zeros(nrow,ncol); %pre-allocate a mask
    color_mask = zeros(nrow,ncol,3);
    instance_mask = zeros(nrow,ncol);
    %mask_final = [];
    for zz=1:length(xy) %for each region
        fprintf('Processing object # %d \n',zz);
        smaller_x=xy{zz}(:,1); 
        smaller_y=xy{zz}(:,2);

        %make a mask and add it to the current mask
        %this addition makes it obvious when more than 1 layer overlap each
        %other, can be changed to simply an OR depending on application.
        polygon = poly2mask(smaller_x,smaller_y,nrow,ncol);
        binary_mask=binary_mask+zz*(1-min(1,binary_mask)).*polygon;%
        color_mask = color_mask + cat(3, rand*polygon, rand*polygon,rand*polygon);
        instance_mask(polygon==1) = max(instance_mask(:))+1;
        %binary mask for all objects
        %imshow(ditance_transform)
    end
    instance_mask = cast(instance_mask, 'uint16');
end