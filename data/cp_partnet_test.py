import shutil
import os

obj_list = [["38037","03001627","58a1e9909542abbae48dacf1789d97b"],
            ["42213","03001627","b89cbb45476e94a5e65235d5580cc3e0"],
            ["45005","03001627","fdf0fd3b18066cd7e8b2b8dc0c816caf"],
            ["38274","03001627","5b9ebc70e9a79b69c77d45d65dc3714"],
            ["1284","03001627","115b11a77b8d8c3c110a27d1d78196"],
            ["39270","03001627","70f1f85d47c970bb78dd615a59de5f05"],
            ["42569","03001627","c30813d1130b492e81b31fbac7146568"],
            ["40399","03001627","8951c681ee693af213493f4cb10e07b0"],
            ["39194","03001627","6ecec258a1b6fe2a6fee8e2140acec9"],
            ["40047","03001627","7e2ef2a1256f2dc1ebe13e25a6ad0d"],
            ["38637","03001627","64d07a909361ccdd1a8a283df3396be6"],
            ["38947","03001627","6a8f1dd7e0642ca4b367a648e9b173ab"],
            ["37206","03001627","47cd848a5584867b1e8791c225564ae0"],
            ["43310","03001627","d73e46e07bdb3fe75fe4ecea39e8bd40"],
            ["44954","03001627","fcc996c2c8ff394194887ea54f3cfb87"],
            ["762","03001627","10d174a00639990492d9da2668ec34c"],
            ["41831","03001627","a7ae914aba9eb59d84498bc295cd5a4a"],
            ["42586","03001627","c3cfd2188fe7ae365fe4ecea39e8bd40"],
            ["43121","03001627","d2a5b42cf29b753f71a782a4379556c7"],
            ["41527","03001627","a147244346d84fe398e0d1738edd4f19"],
            ["40228","03001627","854f3cc942581aea5af597c14b093f6"],
            ["2728","03001627","2b52cd0ffce12156ccbcb819724fb563"],
            ["39679","03001627","77fbfd2f194ed73975aa7f24a9b6003a"],
            ["40698","03001627","909244db9219fb7c5bb4f4519002140"],
            ["41877","03001627","a8da22b87a249bc9c9bfaa062f2e9d4c"],
            ["39148","03001627","6dfa9675d27b5fc134f6a34e269f5ec1"],
            ["38317","03001627","5d346bdb7db27accf3588493d5c284"],
            ["38554","03001627","632a5ea290b0730c6ad8177a9d42d3c9"],
            ["12394","02818832","a39d775401e716948cbb8bac2032149c"],
            ["12802","02818832","f10984c7255bc5b25519d54a714fac86"],
            ["12139","02818832","75e308dc83c1c1f0bf20c89392e58bb0"],
            ["11196","02818832","37ca2cb730cac30ea42bba87fb4a4a5"],
            ["12482","02818832","a2499fdc5535875087693c0a016606d4"],
            ["12032","02818832","6e5f10f2574f8a285d64ca7820a9c2ca"],
            ["12747","02818832","c2b65540d51fada22bfa21768064df9c"],
            ["11548","02818832","5e930124cbecf7091da03d263d0d0387"],
            ["10873","02818832","1619aa637cbb6e131ba2f806cba87b47"],
            ["11350","02818832","42e4e91343b44d77c3bd24f986301745"],
            ["5861","03593526","770f2adb9756b792c9f016d57db96408"],
            ["4376","03593526","311f29deecdde671295ffb61d1cd3a33"]
            ]
            
            

def copy_file(source_path, destination_path):
    
    shutil.copy(source_path, destination_path)
    print(f"File copied successfully from {source_path} to {destination_path}")


for obj_info in obj_list:
    partnet_id = obj_info[0]
    shapenet_id_1 = obj_info[1]
    shapenet_id_2 = obj_info[2]

    os.makedirs("partnet/"+partnet_id,exist_ok=True)
    os.makedirs("partnet/"+partnet_id+"/models",exist_ok=True)
    os.makedirs("partnet/"+partnet_id+"/point_sample",exist_ok=True)

    source_file = "partnet_origin/"+partnet_id+"/result.json"
    target_file = "partnet/"+partnet_id+"/result.json"
    copy_file(source_file, target_file)

    source_file =  "partnet_origin/"+partnet_id+"/point_sample/sample-points-all-label-10000.txt"
    target_file = "partnet/"+partnet_id+"/point_sample/sample-points-all-label-10000.txt"
    copy_file(source_file, target_file)

    source_file =  "partnet_origin/"+partnet_id+"/point_sample/sample-points-all-pts-label-10000.ply"
    target_file = "partnet/"+partnet_id+"/point_sample/sample-points-all-pts-label-10000.ply"
    copy_file(source_file, target_file)

    source_file =  "shapent_origin/"+shapenet_id_1+"/"+shapenet_id_2+"/models/model_normalized.mtl"
    target_file = "partnet/"+partnet_id+"/models/model_normalized.mtl"
    copy_file(source_file, target_file)

    source_file =  "shapent_origin/"+shapenet_id_1+"/"+shapenet_id_2+"/models/model_normalized.obj"
    target_file = "partnet/"+partnet_id+"/models/model_normalized.obj"
    copy_file(source_file, target_file)