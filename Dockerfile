FROM mambaorg/micromamba:1.5.7 as build

USER root

USER $MAMBA_USER
ARG PATH="$PATH:/opt/conda/bin"
WORKDIR /home/$MAMBA_USER

COPY env.linux.lock.yml  .

RUN micromamba install -n base --file env.linux.lock.yml

COPY --chown=$MAMBA_USER vitssm ./vitssm
COPY --chown=$MAMBA_USER run_configs ./run_configs

COPY --link vitssm/ ./vitssm
COPY --link run_configs/ ./run_configs

ENV PATH="$PATH:/opt/conda/bin"
ENV PYTHONPATH="/home/$MAMBA_USER/vitssm"
