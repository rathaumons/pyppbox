# Welcome to pyppbox !

## 🐍📦 pyppbox = Python  + People + Toolbox 

* Yes, pyppbox is an open-source Python toolbox which was originally made for the PoseTReID framework. 
* This toolbox features 3 main modules: People Detection module, People Tracking module, and People Re-identifying module. These modules support a bunch of well-known people detectors, trackers, and re-identificators which can be used separately or put together with only a few lines of code. 
* **pyppbox** also supports real-time online and offline evaluation on [PoseTReID datasets](https://github.com/rathaumons/PoseTReID_DATASET).
* The initial version of pyppbox is also intergrated with GUI for easy demo and config. 

![alt text](https://raw.githubusercontent.com/rathaROG/screenshot/master/pyppbox/pyppbox_launchGUI.png)

## ⚽ Comparison Results

* Comparisions on PoseTReID datasets (Check our pre-printed paper: http://arxiv.org/abs/2205.10086)
* [Click here for raw results in the paper](https://drive.google.com/open?id=13pVqKKd0mtoAaVQh1USxOwZwxg4HmzyQ)

<img src="https://raw.githubusercontent.com/rathaROG/screenshot/master/pyppbox/pyppbox_res001n.png">

## ⚙️ Requirements

* This is the very first version of pyppbox, and we currently support only **Windows** due to the incompatibility of some modules on other platforms with the latest hardware.

* This current version supports only **Python 3.9.x** and **CUDA 11.3.x** & **CUDNN 8.3.x**. If you wish to build pyppbox on different version of Python and CUDA & CUDNN, etc., you must build the corresponding required packages such as [opencv-contrib-python (With CVDNN)](https://github.com/rathaumons/pyppbox/blob/main/requirements/cust/opencv_contrib_python-4.5.5-cp39-cp39-win_amd64.whl) and the [customized torchreid](https://github.com/rathaumons/pyppbox/blob/main/requirements/cust/torchreid-1.4.0-cp39-cp39-win_amd64.whl) (Check out the [customized torchreid repo here](https://github.com/rathaumons/torchreid-for-pyppbox)).

### Install requirements
* Optional: If you prefer conda, you can create a virtual conda **Python 3.9** enviroment of your choice.
```
conda create --name pyppbox_env python=3.9
conda activate pyppbox_env
```
* Simply run the install_pippackages.cmd
```
cd requirements
install_pippackages.cmd
```

### Verify the requirements
* Simply run the testme.cmd
```
testme.cmd
```
* If there is no error, then you're all good and ready to go.

## 🚀 Setup pyppbox

### Option 1: Use the prebuilt WHL file
* Get the latest WHL from here: https://drive.google.com/open?id=11FEf50FEOYpz1FoHFNwIFea0RXbQzSlo
```
pip install pyppbox-1.0b5-cp39-cp39-win_amd64.whl
```

### Option 2: Build your own pyppbox (Comming soon!)
* Build WHL using the setup.py
```
pip install scikit-build
pip wheel . --verbose
```
* Then simply install your newly created WHL file

### Quick Test
* On your terminal or CMD:
```
python
import pyppbox
pyppbox.launchGUI()
```
* Now you should see [the GUI of pyppbox for easy demo](https://raw.githubusercontent.com/rathaROG/screenshot/master/pyppbox/pyppbox_launchGUI.png). Just hit the "LAUNCH" button!

## 📝 Documentation? 

* **[COMING SOON ⌛](https://github.com/rathaumons/pyppbox)**
* Meanwhile you can check [the examples here](https://github.com/rathaumons/pyppbox/tree/main/examples) ! 

## 🔗 Citation

* Extension of PoseTReID paper (Pre-printed on ARXIV):
```
@misc{https://doi.org/10.48550/arxiv.2205.10086,
  doi = {10.48550/ARXIV.2205.10086},
  url = {https://arxiv.org/abs/2205.10086},
  author = {Siv, Ratha and Mancas, Matei and Gosselin, Bernard and Valy, Dona and Sreng, Sokchenda},
  title = {People Tracking and Re-Identifying in Distributed Contexts: Extension of PoseTReID},
  publisher = {arXiv},
  year = {2022},
```

* Original PoseTReID paper:
```
@INPROCEEDINGS{ptreid9271712,
  author={Siv, Ratha and Mancas, Matei and Sreng, Sokchenda and Chhun, Sophea and Gosselin, Bernard},
  booktitle={2020 12th International Conference on Information Technology and Electrical Engineering (ICITEE)}, 
  title={People Tracking and Re-Identifying in Distributed Contexts: PoseTReID Framework and Dataset}, 
  year={2020},
  pages={323-328},
  doi={10.1109/ICITEE49829.2020.9271712}}
```
