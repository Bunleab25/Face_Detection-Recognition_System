# Face Detection & Recognition System - Project Checklist

## Pre-Push Checklist

Before pushing to GitHub, make sure you've completed these steps:

### ✅ Code Quality
- [ ] All Python files follow PEP 8 style guidelines
- [ ] No sensitive data (API keys, passwords) in code
- [ ] No hardcoded absolute paths (use relative paths)
- [ ] All commented code is removed or properly documented
- [ ] Type hints added to function signatures

### ✅ Files & Structure
- [ ] README.md is up to date
- [ ] LICENSE file exists
- [ ] .gitignore properly excludes large files/datasets
- [ ] Requirements files are complete and tested
- [ ] Documentation in docs/ folder is current

### ✅ Dependencies
- [ ] requirements.txt includes all core dependencies
- [ ] requirements-gpu.txt has correct CUDA version
- [ ] requirements-cpu.txt for CPU-only setup
- [ ] requirements_fastapi.txt for web application
- [ ] No unnecessary dependencies

### ✅ Testing
- [ ] Code runs without errors on fresh environment
- [ ] Tested on both CPU and GPU (if available)
- [ ] Sample scripts execute successfully
- [ ] Web application starts without issues

### ✅ Data & Models
- [ ] Large datasets are in .gitignore
- [ ] Model weights handling is documented
- [ ] Sample data is provided (if applicable)
- [ ] Dataset licenses are acknowledged

### ✅ Documentation
- [ ] README has clear installation instructions
- [ ] Usage examples are provided
- [ ] API documentation is complete (for web app)
- [ ] Known issues are documented
- [ ] Contributing guidelines exist

### ✅ Git Configuration
- [ ] .gitignore is properly configured
- [ ] .gitattributes for LFS (if using large files)
- [ ] Commit messages are clear and descriptive
- [ ] Branch is clean (no merge conflicts)

### ✅ GitHub Specific
- [ ] Repository name is appropriate
- [ ] Repository description is added
- [ ] Topics/tags are set for discoverability
- [ ] Issues are enabled
- [ ] Wiki is configured (optional)

## Initial Push Commands

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Face Detection & Recognition System"

# Add remote repository (replace with your GitHub repo URL)
git remote add origin https://github.com/yourusername/face_detector.git

# Push to GitHub
git push -u origin main
```

## Post-Push Tasks

- [ ] Verify all files are visible on GitHub
- [ ] Check README renders correctly
- [ ] Test clone and setup on fresh machine
- [ ] Add repository description and topics
- [ ] Create release tags (optional)
- [ ] Set up GitHub Actions (optional)
- [ ] Enable GitHub Pages for documentation (optional)

## Recommended GitHub Settings

1. **Repository Settings**
   - Make repository public/private as needed
   - Add description and website
   - Add topics: `face-detection`, `face-recognition`, `yolo`, `arcface`, `pytorch`

2. **Branch Protection** (for main branch)
   - Require pull request reviews
   - Require status checks to pass

3. **Issues**
   - Enable issue templates
   - Set up labels (bug, enhancement, documentation, etc.)

4. **Actions**
   - Enable GitHub Actions for CI/CD
   - Review workflow files

## Large Files Handling

If you have large model files (>100MB), consider:

1. **Git LFS (Large File Storage)**
```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes
```

2. **External Hosting**
   - Upload models to Google Drive, Dropbox, or Hugging Face
   - Add download links to README
   - Create download script

3. **Releases**
   - Add model weights as release assets
   - Users can download from releases page

## Troubleshooting

**Problem: Repository size too large**
- Check if large files are accidentally tracked
- Use `git lfs` for model files
- Move datasets to external storage

**Problem: Sensitive data committed**
- Use `git filter-branch` or BFG Repo-Cleaner
- Update all passwords/API keys immediately
- Add to .gitignore

**Problem: Dependencies not installing**
- Verify requirements.txt versions
- Test in clean virtual environment
- Document OS-specific issues

---

**Last Updated:** January 8, 2026
