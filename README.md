# mujoco-py_rendering
To test the image rendering performance & installation of mujoco-py (by openai)

## Setting up instructions

  - Clone the repository
  ```
  git clone --recursive https://github.com/ShahRutav/mujoco-py_rendering.git
  ```
  - Install mujoco-py (detailed instructions are provided here https://github.com/ShahRutav/mujoco-py_rendering#installing-mujoco-py-with-gpu-support)
  - `pip install -r requirements.txt`
  - `pip install -e mjrl/.`

TODO

## Running Instructions

```
python speedtest.py 
       --env_name <env_name> 
       -e <number of episodes> 
       -p <path to the policy> 
       -c <name of the camera>
```

## Installing mujoco-py with GPU support
```
git clone https://github.com/openai/mujoco-py.git

cd mujoco-py/

vim mujoco_py/builder.py > Go to line 74 (https://github.com/openai/mujoco-py/blob/d73ce6e91d096b74da2a2fcb0a4164e10db5f641/mujoco_py/builder.py#L74) > Change CPU to GPU  # A simple hack that worked for me.

pip install -r requirements.txt

pip install -r requirements.dev.txt

python setup.py install

python setup.py develop

```

## Installing mujoco-py with CPU support
```
pip install -U 'mujoco-py<2.1,>=2.0'
```
### Common Issues

- **Some common issues** faced while installing & their solutions are summarized here : 
https://github.com/aravindr93/mjrl/tree/master/setup#installation 


- **For GLEW related issue** : 
https://stackoverflow.com/questions/15852417/compiling-opengl-program-gl-glew-h-missing

  - Error : 
    ```
    gl/eglshim.c:4:21: fatal error: GL/glew.h: No such file or directory
 
    #include <GL/glew.h>
    ```
  - Simple hack : https://github.com/openai/mujoco-py/issues/218 

    edit `mujoco-py/gl/eglshim.c` and change `#include <GL/glew.h>` to `</home/user/miniconda/include/GL/glew.h>` then again execute `python setup.py install` now this time hopefully things will go according to plan and no errors will occur.


- For **osmea related** dependency issue : 
https://github.com/openai/mujoco-py/issues/96
