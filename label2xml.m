function label2xml
clear; close all; clc;
%% WRITE SPECLABEL IMAGE FORMAT TO XML FILE FOR LEARNING
% EXAMPLE FOLDER INCLUDES
% - mc_label
%   - 7-8
%       - data
%           - ..
%   - 23-30
%       - data
%           - ..
figure;
dir_path        = 'C:\Users\азат\Desktop\mc_label';
filename_list   = getFileList(dir_path);

for k = 1:length(filename_list)
    image_name = filename_list(k).origin;
    label      = filename_list(k).label;
    
    image = imread(image_name);
    imshow(image);
    [r,c,n] = size(image);
    label = load(label);
    [box,num] = getBox(label);
    drawnow;
    %saveXml();    
end
end
%% END
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% XML FILE EXAMPLE
% <annotation>
%     <folder>images</folder>
%     <filename>20170121_145942_011_left00070.png</filename>
%     <path>/home/ramil/DL/data/9-11/data/images/20170121_145942_011_left00070.png</path>
%     <source>
%         <database>Unknown</database>
%     </source>
%     <size>
%         <width>1920</width>
%         <height>1080</height>
%         <depth>3</depth>
%     </size>
%     <segmented>0</segmented>
%     <object>
%         <name>switch</name>
%         <pose>Unspecified</pose>
%         <truncated>0</truncated>
%         <difficult>0</difficult>
%         <bndbox>
%             <xmin>815</xmin>
%             <ymin>734</ymin>
%             <xmax>1223</xmax>
%             <ymax>1071</ymax>
%         </bndbox>
%     </object>
% </annotation>

function saveXml(dirname)
% <annotation verified="yes">
        docNode = com.mathworks.xml.XMLUtils.createDocument('annotation');
        docRootNode = docNode.getDocumentElement;
        docRootNode.setAttribute('verified','yes');
        % 	<folder>images</folder>
        thisElement = docNode.createElement('folder');
        thisElement.appendChild(docNode.createTextNode(dirname));
        docRootNode.appendChild(thisElement);
        % 	<path>/Users/datitran/Desktop/raccoon/images/raccoon-10.png</path>
        thisElement = docNode.createElement('path');
        thisElement.appendChild(docNode.createTextNode([dir_path image_name]));
        docRootNode.appendChild(thisElement);
        % 	<source>
        % 		<database>Unknown</database>
        % 	</source>
        thisElement   = docNode.createElement('source');
        docRootNode.appendChild(thisElement);
        curr_node     = docNode.createElement('database');
        curr_node.appendChild(docNode.createTextNode('Unknown'));
        thisElement.appendChild(curr_node);
        % 	<size>
        % 		<width>450</width>
        % 		<height>495</height>
        % 		<depth>3</depth>
        % 	</size> 
        thisElement   = docNode.createElement('size');
        docRootNode.appendChild(thisElement);
        curr_node     = docNode.createElement('width');
        curr_node.appendChild(docNode.createTextNode(num2str(c)));
        thisElement.appendChild(curr_node);
        curr_node     = docNode.createElement('height');
        curr_node.appendChild(docNode.createTextNode(num2str(r)));
        thisElement.appendChild(curr_node);
        curr_node     = docNode.createElement('depth');
        curr_node.appendChild(docNode.createTextNode(num2str(n)));
        thisElement.appendChild(curr_node);
        % 	<segmented>0</segmented>
        thisElement = docNode.createElement('segmented');
        thisElement.appendChild(docNode.createTextNode('0'));
        docRootNode.appendChild(thisElement);
        %        <object>
        % 		<name>raccoon</name>
        % 		<pose>Unspecified</pose>
        % 		<truncated>0</truncated>
        % 		<difficult>0</difficult>
        % 		<bndbox>
        % 			<xmin>130</xmin>
        % 			<ymin>2</ymin>
        % 			<xmax>446</xmax>
        % 			<ymax>488</ymax>
        % 		</bndbox>
        % 	</object>
        for number_obj = 1:1
            thisElement   = docNode.createElement('object');
            docRootNode.appendChild(thisElement);
            curr_node     = docNode.createElement('name');
            curr_node.appendChild(docNode.createTextNode('rail_ways'));
            thisElement.appendChild(curr_node);
            curr_node     = docNode.createElement('pose');
            curr_node.appendChild(docNode.createTextNode('Unspecified'));
            thisElement.appendChild(curr_node);
            curr_node     = docNode.createElement('truncated');
            curr_node.appendChild(docNode.createTextNode('0'));
            thisElement.appendChild(curr_node);
            curr_node     = docNode.createElement('difficult');
            curr_node.appendChild(docNode.createTextNode('0'));
            thisElement.appendChild(curr_node);
            % 		<bndbox>
            % 			<xmin>130</xmin>
            % 			<ymin>2</ymin>
            % 			<xmax>446</xmax>
            % 			<ymax>488</ymax>
            % 		</bndbox>
            curr_node     = docNode.createElement('bndbox');
            thisElement.appendChild(curr_node);
            curr_node_last     = docNode.createElement('xmin');
            curr_node_last.appendChild(docNode.createTextNode('130'));
            curr_node.appendChild(curr_node_last);
            curr_node_last     = docNode.createElement('ymin');
            curr_node_last.appendChild(docNode.createTextNode('2'));
            curr_node.appendChild(curr_node_last);
            curr_node_last     = docNode.createElement('xmax');
            curr_node_last.appendChild(docNode.createTextNode('446'));
            curr_node.appendChild(curr_node_last);
            curr_node_last     = docNode.createElement('ymax');
            curr_node_last.appendChild(docNode.createTextNode('488'));
            curr_node.appendChild(curr_node_last);
        end
        xmlFileName = 'roads_way.xml';
        xmlwrite(xmlFileName,docNode);
        type(xmlFileName);
end

function [box,num] = getBox(label)
    box = 1;
    num = 1;
    for k = 1:length(label.L)
        if (label.L{k}.class == 20)
            uv = label.L{k}.polyline;
            hold on;
            plot(uv(:,1),uv(:,2),'dr');
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function filename_list   = getFileList(dir_path)
% EXAMPLE FOLDER INCLUDES
% - mc_label
%   - 7-8
%       - data
%           - images
%               - images.png
%           - instances
%               - images.png
%           - labels
%               - images.mat
%           - segments
%               - images.png
dir_list = getPath(dir_path);
number = 1;
for k = 1:length(dir_list)
    full_name = [dir_list(k).name '\data\images'];
    current_files_list = dir(full_name);
    for num = 1:length(current_files_list)
        if (~strcmp(current_files_list(num).name,'.') && ...
            ~strcmp(current_files_list(num).name,'..'))
            f_name = [full_name '\' current_files_list(num).name];
            lab_name = strrep(f_name,'images','labels');
            lab_name = strrep(lab_name,'.png','.mat');
            
            filename_list(number).origin = f_name;
            filename_list(number).label  = lab_name;
            number = number + 1;
        end
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function list = getPath(path_name)
    num = 1;
    dir_list  = dir(path_name);
    for number = 1:length(dir_list)
        if (~strcmp(dir_list(number).name,'.') && ...
            ~strcmp(dir_list(number).name,'..'))
            list(num).name = [path_name '\' dir_list(number).name];
            num = num + 1;
        end
    end
end
