# Quick Reference - Git Commands

## 🚀 First Time Setup

### Initialize and Push to GitHub

```bash
# 1. Create repository on GitHub first, then:

# 2. Initialize local repository
git init
git add .
git commit -m "Initial commit: Face Detection & Recognition System"

# 3. Connect to GitHub (replace with your URL)
git remote add origin https://github.com/yourusername/face_detector.git
git branch -M main
git push -u origin main
```

## 📝 Daily Git Workflow

### Check Status
```bash
git status                    # See what changed
git diff                      # See detailed changes
git log --oneline            # View commit history
```

### Stage and Commit Changes
```bash
git add .                     # Stage all changes
git add file.py              # Stage specific file
git commit -m "Description"   # Commit with message
```

### Push and Pull
```bash
git push                      # Push to GitHub
git pull                      # Pull from GitHub
git fetch                     # Fetch without merging
```

## 🌿 Branch Management

```bash
# Create and switch to new branch
git checkout -b feature-name

# Switch branches
git checkout main
git checkout feature-name

# List branches
git branch                    # Local branches
git branch -r                # Remote branches
git branch -a                # All branches

# Merge branch
git checkout main
git merge feature-name

# Delete branch
git branch -d feature-name    # Delete local
git push origin --delete feature-name  # Delete remote
```

## 🔄 Undo Changes

```bash
# Discard changes in file
git checkout -- file.py

# Unstage file
git reset HEAD file.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Revert a commit (creates new commit)
git revert <commit-hash>
```

## 🏷️ Tags and Releases

```bash
# Create tag
git tag v1.0.0
git tag -a v1.0.0 -m "Version 1.0.0"

# Push tags
git push origin v1.0.0       # Specific tag
git push --tags              # All tags

# List tags
git tag

# Delete tag
git tag -d v1.0.0           # Local
git push origin --delete v1.0.0  # Remote
```

## 🔍 Viewing History

```bash
# View commits
git log                      # Full log
git log --oneline           # Compact view
git log --graph             # With branch graph
git log -p                  # With diffs

# View specific file history
git log -- file.py

# View changes in commit
git show <commit-hash>
```

## 🔗 Remote Management

```bash
# View remotes
git remote -v

# Add remote
git remote add origin <url>

# Change remote URL
git remote set-url origin <new-url>

# Remove remote
git remote remove origin
```

## 🔧 Configuration

```bash
# Set user info (global)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set user info (local repository)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# View config
git config --list
git config user.name

# Set default editor
git config --global core.editor "code --wait"

# Set default branch name
git config --global init.defaultBranch main
```

## 📦 Git LFS (Large Files)

```bash
# Install Git LFS
git lfs install

# Track file types
git lfs track "*.pt"
git lfs track "*.pth"

# Track specific file
git lfs track "model.pt"

# View tracked files
git lfs ls-files

# View tracking patterns
cat .gitattributes
```

## 🚨 Emergency Commands

### Remove file from Git but keep locally
```bash
git rm --cached file.txt
git commit -m "Remove file from Git"
```

### Remove sensitive data from history
```bash
# Use BFG Repo-Cleaner (recommended)
# Download from: https://rtyley.github.io/bfg-repo-cleaner/

bfg --delete-files sensitive.txt
bfg --replace-text passwords.txt
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Fix "detached HEAD"
```bash
git checkout main
# or create a branch from detached state
git checkout -b new-branch
```

### Sync fork with original
```bash
git remote add upstream <original-repo-url>
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## 📊 Useful Aliases

Add to your `.gitconfig`:

```bash
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual 'log --graph --oneline --all'
```

Then use:
```bash
git co main              # checkout
git br                   # branch
git ci -m "message"      # commit
git st                   # status
git visual               # pretty log
```

## 🔐 Authentication

### HTTPS with Token (Recommended)
```bash
# Generate token at: https://github.com/settings/tokens
# Use token as password when prompted
git clone https://github.com/username/repo.git
```

### SSH Setup
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Copy public key
# Windows
cat ~/.ssh/id_ed25519.pub | clip
# Linux/Mac
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings > SSH and GPG keys
# Test connection
ssh -T git@github.com
```

## 📱 GitHub CLI (gh)

```bash
# Install: https://cli.github.com/

# Authenticate
gh auth login

# Clone repo
gh repo clone username/repo

# Create repo
gh repo create

# View issues
gh issue list

# Create pull request
gh pr create

# View pull requests
gh pr list
```

## 🎯 Project-Specific Tips

### Before First Push
```bash
# Check what will be pushed
git status
git ls-files --others --exclude-standard  # Untracked files

# Check file sizes
git ls-files | xargs ls -lh | sort -k5 -h -r | head -20

# Test .gitignore
git check-ignore -v *
```

### Regular Commits
```bash
# Meaningful commit messages
git commit -m "feat: add face alignment script"
git commit -m "fix: resolve CUDA memory issue"
git commit -m "docs: update README with API examples"
git commit -m "refactor: optimize detection pipeline"
```

### Commit Message Conventions
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Formatting
- `refactor:` - Code restructuring
- `test:` - Adding tests
- `chore:` - Maintenance

---

## 📚 Learn More

- Official Git Documentation: https://git-scm.com/doc
- GitHub Guides: https://guides.github.com/
- Pro Git Book: https://git-scm.com/book/en/v2
- Interactive Tutorial: https://learngitbranching.js.org/

---

**Keep this file handy for quick reference!** 📌
