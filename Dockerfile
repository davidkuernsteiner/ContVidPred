# Use the official miniconda3 image as the base image
FROM mambaorg/micromamba:1.4.2

# Set the working directory
WORKDIR /launch

# Copy the environment file
COPY --link environment.yml ./

# Install the environment using Micromamba
RUN micromamba install -y -n base -f /launch/environment.yml && \
    micromamba clean --all --yes

# Install additional Python packages using pip3 with increased timeout and retries
RUN micromamba run -n base pip install --default-timeout=1000 torch torchvision torchaudio torchmetrics && \
    micromamba run -n base pip install --default-timeout=1000 -U xformers --index-url https://download.pytorch.org/whl/cu121

# Make sure the environment is activated by default
ENV MAMBA_DOCKERFILE_ACTIVATE=1

COPY --link vitssm/ data/ configs/  ./

SHELL ["micromamba", "run", "-n", "base", "/bin/bash", "-c"]
SHELL ["python", "vitssm/execute_job.py", "-bc configs/base_config.yml", "-rc configs/config_run_1.yml"]
