import shutil
import os

obj_list = [["3069","03001627","32d9f69ef8ebb1778a514cac7cb18507"],
            ["37825","03001627","53eaa7cca72c84f6cacd67ce2c72c02e"],
            ["43941","03001627","e908c290fb506f2e19fb4103277a6b93"],
            ["40503","03001627","8bb3a13d45c5337d20e3ea5765d7edb"],
            ["37986","03001627","5840d369b3e0e40f7c4ed45ce654123"],
            ["3305","03001627","383bac847e38daa0e8dd9b2c07157c63"],
            ["43733","03001627","e39df7339552595febad4f49b26ec52"],
            ["39612","03001627","7710ecf956701938b40f0ac0fb9a650d"],
            ["42242","03001627","b960544cfd0ff09f26b2c6e6d8c1e1ab"],
            ["36520","03001627","3d3b7f63f5525b1ae37f5a622d383617"],
            ["40427","03001627","895be5f65513a7d09a8ef44e1d2c5b75"],
            ["37738","03001627","51c858aec859bafad1e274d497440a3e"],
            ["44938","03001627","fc6129a9310ba34c645311c54e2f9bdc"],
            ["36545","03001627","3d9dce1953180fe6f9c9f9697d1ec60"],
            ["41211","03001627","9a68fc6d001c4ceadc75c30c88b2f7a9"],
            ["43721","03001627","e32ee21232d2d5604747ada1cb39a749"],
            ["43075","03001627","d1f76ed6072b9332ee558d9fec5dbe41"],
            ["41936","03001627","aae036d8ebdc472535836c728d324152"],
            ["41240","03001627","9adb6a665f99addc8a4fd70ea4c8b4d9"],
            ["38607","03001627","64067f7029087ee07eaeab1f0c9120b7"],
            ["43779","03001627","e4ac472d21d43d3258db0ef36af1d3c5"],
            ["40206","03001627","82d8391c62e197161282d4d9178caa92"],
            ["2419","03001627","20e1bdd54d4082097962800be79c6e52"],
            ["37861","03001627","54f13fbf4a274267a50b88953d263a42"],
            ["43148","03001627","d350f40e5f04360565ba78ad9601cf1b"],
            ["39163","03001627","6e53d494768386ca8579483a049f2a91"],
            ["12159","02818832","7c8eb4ab1f2c8bfa2fb46fb8b9b1ac9f"],
            ["12517","02818832","ba690c29eb60a5601112a9ee83a73f73"],
            ["12524","02818832","c758919a1fbe3d0c9cef17528faf7bc5"],
            ["12493","02818832","af2e51ff34f1084564dabb9b8161e0a7"],
            ["11338","02818832","57764cebd92102552ea98d69e91ba870"],
            ["12203","02818832","845942aee9bee62b9f2349486c570dd4"],
            ["12330","02818832","9fb6014c9944a98bd2096b2fa6f98cc7"],
            ["12319","02818832","9daa02ae72e74b3f6bb2e9a62ccd177a"],
            ["12452","02818832","ac97ffc6b6ef238382c3b56998e08a4c"],
            ["11749","02818832","6256db826fbb31add7e7281b421bca5"],
            ["12083","02818832","734d564442cc75472b0c00d36a59e875"],
            ["11098","02818832","2e149e4f08f7cbdfed4d676215f46734"],
            ["11108","02818832","2cc7c1f0c903278bc3bd24f986301745"],
            ["11570","03642806","cb090bd99ed76f9689661310be87a70d"],
            ["26432","04379243","9c4dfafdbd7f9b76c955e5ed03ef3a2f"]
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