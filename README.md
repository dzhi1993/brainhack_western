## brainhack_western

Brainhack 2019 in Western University, Diedrichsen Lab (Ongoing)

This project aims to show cortical surface flatmap using python-based opengl programming. The first version features to allow users to change underlay, add multiple overlays and borders, etc. Eventually, we want to make it as a handy package for lab use to achieve several neuroimaging tasks and involves the 3D volume viewer along with the flatmap as an interactive image viewer, similar to the [AtlasViewer] (http://www.diedrichsenlab.org/imaging/AtlasViewer/viewer.html).

### Installation
Install OpenGL library of python version through standard pip installation:

	pip install PyOpenGL PyOpenGL_accelerate


Or you can install the source repository by branching/cloning and running setup.py:	

	git clone https://github.com/mcfletch/pyopengl.git
	cd pyopengl
	python setup.py develop
	cd accelerate
	python setup.py develop


Please follow the instruction in the raw [pyopengl](https://github.com/mcfletch/pyopengl) repository 

Install below dependencies:

	pip install nibabel numpy matplotlib glfw


### Running the code

the default setting has two overlays (test.func.gii, test2.func.gii), which can be found in onlineAtlas folder.

	python main.py


### Contrast data

All data stored in onlineAtlas folder 

The flatmap topology file is `fs_LR.164k.L.flat.surf.gii`

The default left hemisphere underlay is `fs_LR.164k.LR.sulc.dscalar.nii`

The borders information can be loaded via `onlineAtlas/fs_LR.164k.L.border-CS.func.gii`, `onlineAtlas/fs_LR.164k.L.border-IPS.func.gii`, `onlineAtlas/fs_LR.164k.L.border-PoCS.func.gii`, `onlineAtlas/fs_LR.164k.L.border-SF.func.gii`.

The default loaded overlays are `test.func.gii` and `test2.func.gii`