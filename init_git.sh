#!/bin/bash
# Git Initialization Script for Face Detector Project (Linux/Mac)
# This script helps initialize git and prepare for first push to GitHub

echo "========================================"
echo "  Face Detector - Git Setup"
echo "========================================"
echo ""

# Check if git is installed
echo "Checking Git installation..."
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    echo "   Ubuntu/Debian: sudo apt-get install git"
    echo "   macOS: brew install git"
    exit 1
fi

git_version=$(git --version)
echo "✓ $git_version"
echo ""

# Check if already a git repository
if [ -d ".git" ]; then
    echo "⚠ This is already a Git repository."
    read -p "Do you want to reinitialize? (y/n): " reinit
    if [ "$reinit" != "y" ]; then
        echo "Skipping initialization..."
    else
        rm -rf .git
        git init
        echo "✓ Git repository reinitialized"
    fi
else
    echo "Initializing Git repository..."
    git init
    echo "✓ Git repository initialized"
fi

echo ""

# Configure git user (if not already configured)
userName=$(git config user.name)
userEmail=$(git config user.email)

if [ -z "$userName" ]; then
    echo "Git user not configured. Let's set it up:"
    read -p "Enter your name: " name
    git config user.name "$name"
    echo "✓ Name set to: $name"
fi

if [ -z "$userEmail" ]; then
    read -p "Enter your email: " email
    git config user.email "$email"
    echo "✓ Email set to: $email"
fi

echo ""

# Show current configuration
echo "Current Git Configuration:"
echo "  Name:  $(git config user.name)"
echo "  Email: $(git config user.email)"

echo ""

# Check .gitignore
if [ -f ".gitignore" ]; then
    echo "✓ .gitignore file exists"
else
    echo "⚠ .gitignore file not found!"
fi

echo ""

# Stage all files
echo "Staging files..."
git add .

echo ""

# Show status
echo "Git Status:"
git status --short

echo ""

# Create initial commit
read -p "Create initial commit? (y/n): " createCommit
if [ "$createCommit" == "y" ]; then
    read -p "Enter commit message (or press Enter for default): " commitMessage
    if [ -z "$commitMessage" ]; then
        commitMessage="Initial commit: Face Detection & Recognition System"
    fi
    
    git commit -m "$commitMessage"
    echo "✓ Initial commit created"
fi

echo ""

# Add remote repository
read -p "Add GitHub remote repository? (y/n): " addRemote
if [ "$addRemote" == "y" ]; then
    echo ""
    echo "GitHub Repository URL examples:"
    echo "  HTTPS: https://github.com/username/face_detector.git"
    echo "  SSH:   git@github.com:username/face_detector.git"
    echo ""
    
    read -p "Enter your GitHub repository URL: " remoteUrl
    
    # Check if remote already exists
    existingRemote=$(git remote get-url origin 2>/dev/null)
    if [ -n "$existingRemote" ]; then
        echo "⚠ Remote 'origin' already exists: $existingRemote"
        read -p "Update remote URL? (y/n): " updateRemote
        if [ "$updateRemote" == "y" ]; then
            git remote set-url origin "$remoteUrl"
            echo "✓ Remote URL updated"
        fi
    else
        git remote add origin "$remoteUrl"
        echo "✓ Remote repository added"
    fi
fi

echo ""

# Rename branch to main (if needed)
currentBranch=$(git branch --show-current)
if [ "$currentBranch" != "main" ]; then
    read -p "Rename branch from '$currentBranch' to 'main'? (y/n): " renameBranch
    if [ "$renameBranch" == "y" ]; then
        git branch -M main
        echo "✓ Branch renamed to 'main'"
    fi
fi

echo ""

# Push to GitHub
read -p "Push to GitHub now? (y/n): " push
if [ "$push" == "y" ]; then
    echo "Pushing to GitHub..."
    if git push -u origin main; then
        echo "✓ Successfully pushed to GitHub!"
    else
        echo "❌ Push failed. Please check:"
        echo "   1. GitHub repository exists"
        echo "   2. You have write access"
        echo "   3. Authentication is configured"
        echo "   4. Remote URL is correct"
    fi
fi

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Visit your GitHub repository"
echo "  2. Add description and topics"
echo "  3. Review CHECKLIST.md for additional tasks"
echo ""
echo "Useful commands:"
echo "  git status              - Check current status"
echo "  git add .               - Stage all changes"
echo "  git commit -m 'msg'     - Commit changes"
echo "  git push                - Push to GitHub"
echo "  git pull                - Pull from GitHub"
echo ""
