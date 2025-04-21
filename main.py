
from Src.cli import main

if __name__ == "__main__":
    main()

# python main.py -c /home/datasets/ZOUbohao/Proj1-Deepurify/Data_marine/1102218.contigs.fasta -b /home/datasets/ZOUbohao/Proj1-Deepurify/Data_marine/1102218.sorted.bam -o ../DeeperBin-Real-data/marine-1102218-final-bins -temp ../DeeperBin-Real-data/marine-1102218-temp -db ./DeeperBin-DB/ --device cuda:3
# python main.py -c /home/datasets/ZOUbohao/Proj1-Deepurify/Data_freshwater/ERR4195020.contigs.fasta -b /home/datasets/ZOUbohao/Proj1-Deepurify/Data_freshwater/ERR4195020.sorted.bam -o ../DeeperBin-Real-data/freshwater-ERR4195020-final-bins -temp ../DeeperBin-Real-data/freshwater-ERR4195020-temp -db ./DeeperBin-DB/ --device cuda:3
#
# 
# 
# python main.py -c /home/datasets/ZOUbohao/Proj1-Deepurify/Data_plant/SRR10968246.contigs.fasta -b /home/datasets/ZOUbohao/Proj1-Deepurify/Data_plant/SRR10968246.sorted.bam -o ../DeeperBin-Real-data/plant-SRR10968246-final-bins -temp ../DeeperBin-Real-data/plant-SRR10968246-temp -db ./DeeperBin-DB/ --device cuda:3

# python main.py -c ../HD-marine-final-results/CS01.contigs.fasta -b ../HD-marine-final-results/CS01.sorted.bam -o ../DeeperBin-Real-data/HD-CS01-final-bins -temp ../DeeperBin-Real-data/HD-CS01-temp -db ./DeeperBin-DB/ --device cuda:3


