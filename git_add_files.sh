#!/bin/bash

# Function to delete local repository
delete_local_repository() {
  echo "Warning: This will remove the .git directory and all version control history."
  echo "Are you sure you want to delete the local repository? (y/n)"
  read confirmation
  
  if [ "$confirmation" = "y" ] || [ "$confirmation" = "Y" ]; then
    rm -rf .git
    echo "Local repository deleted successfully."
  else
    echo "Operation cancelled."
  fi
}

# Display menu
echo "Choose an option:"
echo "1. Add files to GitHub"
echo "2. Delete local repository"
read option

if [ "$option" = "2" ]; then
  delete_local_repository
  exit 0
fi

# Initialize git repository if it doesn't exist
if [ ! -d .git ]; then
  git init
  echo "Git repository initialized."
else
  echo "Git repository already exists."
fi

# Add all files to staging
git add .
echo "Files added to staging area."

# Commit changes
echo "Enter commit message:"
read commit_message
git commit -m "$commit_message"
echo "Changes committed."

# Check if remote exists, if not, add it
if ! git remote | grep -q "origin"; then
  echo "Enter GitHub repository URL (e.g., https://github.com/username/repo.git):"
  read repo_url
  git remote add origin "$repo_url"
  echo "Remote repository added."
fi

# Push to GitHub
git push -u origin master
echo "Files pushed to GitHub successfully."
