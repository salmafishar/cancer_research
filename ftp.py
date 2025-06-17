import pandas as pd
from ftplib import FTP
import os

def get_files(all_drugs, value_to_pull, output_file_name):
    # create combined df for all drugs
    combined = pd.DataFrame(columns=["gene"])

    all_columns = ["gene"]
    all_columns.append(value_to_pull)

    # set up FTP 
    ftp = FTP("massive-ftp.ucsd.edu")
    ftp.login()
    ftp.cwd("/v06/MSV000093659/other/Dose response data - Jurkat proteome")

    for drug in all_drugs:
        # make a string with the drug name + _ALL.txt 
        drug_file_name = f"{drug}_ALL.txt"
        file_path = "/data/ftp_files"

        # check if the file already exists locally (already downloaded it)
        if not os.path.exists(f"{file_path}/{drug_file_name}"):

            # if it doesn't already exist, use FTP to get that file from the server
            drugPath = f"{drug}/TXTs_Classified"
            ftp.cwd(drugPath)
            
            # download/write the file in binary mode
            with open(drug_file_name, "wb") as file:
            # download the file "RETR fileName"
                ftp.retrbinary(f"RETR {drug_file_name}", file.write)
            
            # move back to the Jurkat proteome directory
            ftp.cwd("../../")

        # load the file as a dataframe
        drugData = pd.read_csv(f"{file_path}/{drug_file_name}", delimiter = "\t") 

        # select the gene and mean intensity columns  
        pulled_data = drugData[all_columns]
        # rename mean intensity column to include the name of the drug
        renamed_data = pulled_data.rename(columns = {f"{value_to_pull}":f"{drug}"})
        # add both of those columns to a combined dataframe (create before the for loop)
        combined = pd.merge(combined, renamed_data, how = "outer", on = "gene")

    ftp.quit()
    combined.set_index("gene")
    combined.to_csv(f"{output_file_name}.csv", index = False)