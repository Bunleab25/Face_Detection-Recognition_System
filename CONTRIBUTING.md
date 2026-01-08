# Contributing to Face Detection & Recognition System

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- System information (OS, Python version, GPU/CPU)
- Error messages or logs

### Suggesting Features

Feature requests are welcome! Please create an issue describing:
- The feature you'd like to see
- Why it would be useful
- Possible implementation approach (optional)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, commented code
   - Follow existing code style
   - Add tests if applicable
   - Update documentation as needed

3. **Test your changes**
   - Ensure all existing tests pass
   - Test on both CPU and GPU if possible
   - Test with sample images/datasets

4. **Commit your changes**
   ```bash
   git commit -m "Add feature: description"
   ```
   
   Use clear commit messages:
   - `feat: add new feature`
   - `fix: resolve bug in module`
   - `docs: update documentation`
   - `refactor: improve code structure`
   - `test: add tests for feature`

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Include screenshots/examples if applicable

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/yourusername/face_detector.git
cd face_detector
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.\.venv\Scripts\Activate.ps1  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt  # if you have CUDA
pip install -r requirements_fastapi.txt  # for web app
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular
- Use meaningful variable names

Example:
```python
def detect_faces(image: np.ndarray, confidence: float = 0.5) -> list:
    """
    Detect faces in an image using YOLO model.
    
    Args:
        image: Input image as numpy array
        confidence: Minimum confidence threshold (0.0-1.0)
    
    Returns:
        List of detected face bounding boxes
    """
    # Implementation
    pass
```

## Testing

- Test your code before submitting
- Include sample test cases
- Verify compatibility with different Python versions (3.8+)

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions
- Update relevant docs in `docs/` folder
- Include usage examples

## Questions?

Feel free to open an issue for any questions or concerns!

Thank you for contributing! 🎉
