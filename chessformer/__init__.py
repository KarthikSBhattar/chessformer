# Create the directory ./checkpoints
import os
os.makedirs("./checkpoints", exist_ok=True)

# Check if the checkpoints are already downloaded
if os.path.exists("./checkpoints/9M"):
    print("Checkpoints already downloaded. Skipping download.")
else:
    os.system("wget https://storage.googleapis.com/searchless_chess/checkpoints/9M.zip")
    os.system("unzip 9M.zip -d checkpoints")
    os.system("rm 9M.zip")