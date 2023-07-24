FROM python:3.10

# Install Java
RUN apt update 

## Pip dependencies
RUN pip install --upgrade pip

#install opencv dependency
RUN apt-get install -y libgl1-mesa-dev


#install opencv
RUN pip install opencv-python numpy scipy matplotlib scikit-image scikit-learn ipython ipykernel ipywidgets pandas sympy nose