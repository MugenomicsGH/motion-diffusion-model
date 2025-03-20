# Create body_models directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "body_models"
Set-Location "body_models"

Write-Host "The smpl files will be stored in the 'body_models/smpl/' folder`n"

# Install gdown if not already installed
python -m pip install gdown

# Download the file
python -m gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"

# Remove existing smpl directory if it exists
if (Test-Path "smpl") {
    Remove-Item -Recurse -Force "smpl"
}

# Extract the zip file
Expand-Archive -Path "smpl.zip" -DestinationPath "."

Write-Host "Cleaning`n"
Remove-Item "smpl.zip"

Write-Host "Downloading done!"

# Go back to original directory
Set-Location ".." 