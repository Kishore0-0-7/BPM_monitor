@echo off
setlocal enabledelayedexpansion

:: Function equivalent for deleting local repository
:delete_local_repository
echo Warning: This will remove the .git directory and all version control history.
set /p confirmation="Are you sure you want to delete the local repository? (y/n): "
if /i "%confirmation%"=="y" (
    if exist .git (
        rmdir /s /q .git
        echo Local repository deleted successfully.
    ) else (
        echo No Git repository found.
    )
) else (
    echo Operation cancelled.
)
exit /b

:: Display menu
echo Choose an option:
echo 1. Add files to GitHub
echo 2. Delete local repository
set /p option="Enter option: "

if "%option%"=="2" (
    call :delete_local_repository
    goto :eof
)

:: Initialize git repository if it doesn't exist
if not exist .git (
    git init
    echo Git repository initialized.
) else (
    echo Git repository already exists.
)

:: Add all files to staging
git add .
echo Files added to staging area.

:: Commit changes
set /p commit_message="Enter commit message: "
git commit -m "%commit_message%"
echo Changes committed.

:: Check if remote exists, if not, add it
git remote | findstr "origin" > nul
if errorlevel 1 (
    set /p repo_url="Enter GitHub repository URL (e.g., https://github.com/username/repo.git): "
    git remote add origin "!repo_url!"
    echo Remote repository added.
)

:: Push to GitHub
git push -u origin master
echo Files pushed to GitHub successfully.
