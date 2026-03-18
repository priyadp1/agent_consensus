import zipfile
with zipfile.ZipFile("data.zip", "r") as i:
    i.extractall("data")

with zipfile.ZipFile("results.zip", "r") as j:
    j.extractall("results")

print("Done extracting file")
    
    
