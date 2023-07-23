FROM python:3.10

# Install Java
RUN apt update 

## Pip dependencies
RUN pip install --upgrade pip