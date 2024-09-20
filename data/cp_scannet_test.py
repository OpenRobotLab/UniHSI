import shutil
import os

obj_list = ["0000_00", "0004_00", "0006_00", "0008_00", "0009_00", "0010_00", "0013_00",
             "0015_00", "0025_00", "0030_00"]
            
            

def copy_file(source_path, destination_path):
    
    shutil.copy(source_path, destination_path)
    print(f"File copied successfully from {source_path} to {destination_path}")


for obj_id in obj_list:
    source_file =  "scannet_origin/scans/"+obj_id+"/"+obj_id+"_vh_clean_2.ply"
    target_file = "scannet/"+obj_id+"_vh_clean_2.ply"
    source_file =  "scannet_origin/scans/"+obj_id+"/"+obj_id+"_vh_clean.aggregation.json"
    target_file = "scannet/"+obj_id+"_vh_clean.aggregation.json"
    source_file =  "scannet_origin/scans/"+obj_id+"/"+obj_id+"_vh_clean_2.0.010000.segs"
    target_file = "scannet/"+obj_id+"_vh_clean_2.0.010000.segs"

    copy_file(source_file, target_file)
