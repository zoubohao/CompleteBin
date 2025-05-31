

import os

from CompleteBin.IO import readMetaInfo


def getScore(
    meta_info
) -> float:
    
    t_score = 0.
    for _, qualityValues in meta_info.items():
        if qualityValues[-1] == "HighQuality":
            score = qualityValues[0] - 4. * qualityValues[1]  + 120.
        elif qualityValues[-1] == "MediumQuality":
            score = qualityValues[0] - 4. * qualityValues[1] + 50.
        else:
            score = 0.
        t_score += score
    return t_score



if __name__ == "__main__":
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-Grid-Search-short-long-ratio/"
    
    file_list = os.listdir(home_path)
    res = []
    for file in file_list:
        if "final-output-bins" in file:
            metainfo_path = os.path.join(home_path, file, "MetaInfo.tsv")
            me, h, m, _ = readMetaInfo(metainfo_path)
            res.append((file, h, m, getScore(me)))
    
    res = list(sorted(res, key=lambda x: x[1], reverse=True))
    for ele in res:
        print(ele)
    


