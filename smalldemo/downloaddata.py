import kagglehub

# Download latest version
path = kagglehub.dataset_download("thedevastator/tinystories-narrative-classification")

print("Path to dataset files:", path)