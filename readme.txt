MS Course Project - Source files
Created by Brandon Percin as part of a MS Course Project

Title:
Denoising Monte Carlo Path-Traced Images using Convolutional Neural Networks


The project tightly couples a rendering engine and a neural network in order to reduce rendering time.

This project is composed of two components that are tightly coupled:
	- Monte Carlo Path Trace Renderer
	- Denoising Neural Network built to reduce rendering time of the renderer


The source code for this project is written entirely in Python. It is composed of 3 source code files:

Renderer.ipynb 
	A Python 3 Jupyter Notebook containing source code for the Monte Carlo-based Path Trace Renderer
	Open the jupyter notebook with jupyter to access source code, documentation, and a visual representation of what the component does.

generate.py
	A Python 3 script that runs the rendering code in the above network, but over the CLI
	This can be used to generate training data for the Neural Network.
	The script has no running parameters, so simply call it over the CLI as 'python3 ./generate.py'
	By default, the script will generate 1000 training examples (pairs of images with varying amounts of samples per pixel)
	This data can then be read by the next component.

train-modular.ipynb
	A Python3 Jupyter Notebook containing all curated Neural Network models, including the most efficient Cognitive Neural Network (and pre-trained models of each)
	Open the jupyter notebook with jupyter to access source code, documentation, and a visual representation of what the component does.

Additionally, the file 'requirements.txt' is an auto-generated file with all required python libraries to run the above source code.
	Simply run "pip3 install -r requirements.txt" when in this directory to automatically install the correct versions of all libraries used.


Additional Files

	Directory 'images'
		- this directory holds images output by the rendering notebook, "renderer.ipynb"
	Directory 'logs'
		- this holds relevant logs for the training of all neural networks featured in the training notebook, "train-modular.ipynb"
	Directory 'saved_models'
		- this holds the architecture and weights of all neural networks in the training notebook
		- additionally, it also holds the output of these neural networks and training graphs in the form of PNG images
	Directory 'scenes'
		- this holds scene files, to be rendered with the "renderer.ipynb" notebook
		- Each file tells where to place camera, spheres, material properties, etc
	Directory 'trainingSamples'
		- this holds the 1000 pairs of generated images for training the neural networks, and the corresponding scene files and depth vectors
		- Additionally, the files 'xRenderTimex.csv' and 'yRenderTimes.csv' record the amount of time it took to render each image.

	ImageGenerationScreenshot.PNG
		- Screenshot from the completion of training sample generation, including the average and total times to create each image