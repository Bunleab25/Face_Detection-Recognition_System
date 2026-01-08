# Git Initialization Script for Face Detector Project
# This script helps initialize git and prepare for first push to GitHub

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Face Detector - Git Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
Write-Host "Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "✓ $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git is not installed. Please install Git first." -ForegroundColor Red
    Write-Host "   Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check if already a git repository
if (Test-Path ".git") {
    Write-Host "⚠ This is already a Git repository." -ForegroundColor Yellow
    $reinit = Read-Host "Do you want to reinitialize? (y/n)"
    if ($reinit -ne "y") {
        Write-Host "Skipping initialization..." -ForegroundColor Yellow
    } else {
        Remove-Item -Recurse -Force .git
        git init
        Write-Host "✓ Git repository reinitialized" -ForegroundColor Green
    }
} else {
    # Initialize git
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✓ Git repository initialized" -ForegroundColor Green
}

Write-Host ""

# Configure git user (if not already configured)
$userName = git config user.name
$userEmail = git config user.email

if (-not $userName) {
    Write-Host "Git user not configured. Let's set it up:" -ForegroundColor Yellow
    $name = Read-Host "Enter your name"
    git config user.name "$name"
    Write-Host "✓ Name set to: $name" -ForegroundColor Green
}

if (-not $userEmail) {
    $email = Read-Host "Enter your email"
    git config user.email "$email"
    Write-Host "✓ Email set to: $email" -ForegroundColor Green
}

Write-Host ""

# Show current configuration
Write-Host "Current Git Configuration:" -ForegroundColor Cyan
Write-Host "  Name:  $(git config user.name)" -ForegroundColor White
Write-Host "  Email: $(git config user.email)" -ForegroundColor White

Write-Host ""

# Check .gitignore
if (Test-Path ".gitignore") {
    Write-Host "✓ .gitignore file exists" -ForegroundColor Green
} else {
    Write-Host "⚠ .gitignore file not found!" -ForegroundColor Yellow
}

Write-Host ""

# Stage all files
Write-Host "Staging files..." -ForegroundColor Yellow
git add .

Write-Host ""

# Show status
Write-Host "Git Status:" -ForegroundColor Cyan
git status --short

Write-Host ""

# Create initial commit
$createCommit = Read-Host "Create initial commit? (y/n)"
if ($createCommit -eq "y") {
    $commitMessage = Read-Host "Enter commit message (or press Enter for default)"
    if ([string]::IsNullOrWhiteSpace($commitMessage)) {
        $commitMessage = "Initial commit: Face Detection & Recognition System"
    }
    
    git commit -m "$commitMessage"
    Write-Host "✓ Initial commit created" -ForegroundColor Green
}

Write-Host ""

# Add remote repository
$addRemote = Read-Host "Add GitHub remote repository? (y/n)"
if ($addRemote -eq "y") {
    Write-Host ""
    Write-Host "GitHub Repository URL examples:" -ForegroundColor Cyan
    Write-Host "  HTTPS: https://github.com/username/face_detector.git" -ForegroundColor Gray
    Write-Host "  SSH:   git@github.com:username/face_detector.git" -ForegroundColor Gray
    Write-Host ""
    
    $remoteUrl = Read-Host "Enter your GitHub repository URL"
    
    # Check if remote already exists
    $existingRemote = git remote get-url origin 2>$null
    if ($existingRemote) {
        Write-Host "⚠ Remote 'origin' already exists: $existingRemote" -ForegroundColor Yellow
        $updateRemote = Read-Host "Update remote URL? (y/n)"
        if ($updateRemote -eq "y") {
            git remote set-url origin $remoteUrl
            Write-Host "✓ Remote URL updated" -ForegroundColor Green
        }
    } else {
        git remote add origin $remoteUrl
        Write-Host "✓ Remote repository added" -ForegroundColor Green
    }
}

Write-Host ""

# Rename branch to main (if needed)
$currentBranch = git branch --show-current
if ($currentBranch -ne "main") {
    $renameBranch = Read-Host "Rename branch from '$currentBranch' to 'main'? (y/n)"
    if ($renameBranch -eq "y") {
        git branch -M main
        Write-Host "✓ Branch renamed to 'main'" -ForegroundColor Green
    }
}

Write-Host ""

# Push to GitHub
$push = Read-Host "Push to GitHub now? (y/n)"
if ($push -eq "y") {
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    try {
        git push -u origin main
        Write-Host "✓ Successfully pushed to GitHub!" -ForegroundColor Green
    } catch {
        Write-Host "❌ Push failed. Please check:" -ForegroundColor Red
        Write-Host "   1. GitHub repository exists" -ForegroundColor Yellow
        Write-Host "   2. You have write access" -ForegroundColor Yellow
        Write-Host "   3. Authentication is configured" -ForegroundColor Yellow
        Write-Host "   4. Remote URL is correct" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Visit your GitHub repository" -ForegroundColor White
Write-Host "  2. Add description and topics" -ForegroundColor White
Write-Host "  3. Review CHECKLIST.md for additional tasks" -ForegroundColor White
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Yellow
Write-Host "  git status              - Check current status" -ForegroundColor Gray
Write-Host "  git add .               - Stage all changes" -ForegroundColor Gray
Write-Host "  git commit -m 'msg'     - Commit changes" -ForegroundColor Gray
Write-Host "  git push                - Push to GitHub" -ForegroundColor Gray
Write-Host "  git pull                - Pull from GitHub" -ForegroundColor Gray
Write-Host ""
