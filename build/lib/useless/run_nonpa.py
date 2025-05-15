
import os

def run_nonp(fastaq_path, output_folder, data_name):
    print(fastaq_path)
    if os.path.exists(fastaq_path) is False:
        print(f"{fastaq_path} not exist.")
        return
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)
    out_reads_path = os.path.join(output_folder, f"{data_name}.fasta")
    # if os.path.exists(out_reads_path) is False:
    print("Nonpa starts to pre-process the fastaq data.")
    cmd1 = f"less {fastaq_path} | paste - - - - | awk " +  "\'BEGIN{FS=\"\\t\"}{print \">\"substr($1,2)\"\\n\"$2}\'"  + f" > {out_reads_path}"
    os.system(cmd1)
    if os.path.exists(os.path.join(output_folder, f"{data_name}.nonp.output.npo")) is False:
        print("Nonp starts to run.")
        # output_folder = os.path.join(input_folder, f"{data_name}.nonp.output")
        cmd2 = f"nonpareil -s {out_reads_path}  -T alignment -f fasta -b {data_name}.nonp.output -t 128 -v 10"
        os.system(cmd2)
        

if __name__ == "__main__":
    pass
