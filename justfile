# Default target: list all available recipes
default:
    @just --list

# Initialize the project and install dependencies
init:
    @echo ""
    @echo "Creating project directories..."
    @mkdir -p src scripts prompts data/input data/output
    @echo "Installing dependencies with uv..."
    @uv sync
    @echo "Project initialized successfully!"
    @echo ""

# Run the main program
run:
    @echo ""
    @uv run src/main.py
    @echo ""

# Remove generated files and cache
clean:
    @echo ""
    @echo "Cleaning up generated files..."
    @rm -rf __pycache__ .pytest_cache .mypy_cache
    @find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    @find . -type f -name "*.pyc" -delete 2>/dev/null || true
    @echo "Cleanup complete!"
    @echo ""
