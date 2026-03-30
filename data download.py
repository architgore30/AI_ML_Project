import kagglehub

# Download latest version
path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")

print("Path to dataset files:", path)

# After downloading: Move dataset to the main folder and rename to dataset.csv