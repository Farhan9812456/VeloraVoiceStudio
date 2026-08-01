FROM yqx267n9na-1784562650138-voice-studio:latest

WORKDIR /workspace

# Relocate virtual environments outside /workspace to avoid shadowing by host bind-mounts
RUN mv /workspace/venvs /venvs

# Downgrade setuptools in the relocated RVC virtual environment to fix the missing pkg_resources bug
RUN /venvs/rvc/bin/python -m pip install --no-cache-dir "setuptools<70"

# Install edge-tts and other application dependencies in the python environment
RUN /venvs/f5tts/bin/python -m pip install --no-cache-dir edge-tts tomli_w noisereduce yt-dlp

# Copy the application source code
COPY . /workspace

# Set environment path to use the relocated virtual environments
ENV PATH="/venvs/f5tts/bin:/venvs/rvc/bin:/usr/local/bin:$PATH"

EXPOSE 7860
CMD ["python", "app.py"]

