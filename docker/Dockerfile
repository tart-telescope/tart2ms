FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install the tartball module
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org tart2ms

# Set metadata labels
LABEL org.opencontainers.image.source="https://github.com/tart-telescope/tartball"
LABEL org.opencontainers.image.description="Prediction code to simulate TART data"
LABEL org.opencontainers.image.licenses="GPL-3.0"

# Default command
CMD ["python", "--version"]
