FROM continuumio/miniconda3

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# Prepare so that env is activated by default
RUN conda init bash
RUN echo "source activate d2v_env" > ~/.bashrc
ENV PATH /opt/conda/envs/d2v_env/bin:$PATH
