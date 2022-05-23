# Welcome to pyppbox !

## 🐍📦 pyppbox = Python  + People + Toolbox 

* Yes, pyppbox is an open-source Python toolbox which is originally made for PoseTReID framework. 
* This toolbox features 3 main moduels: People Detection module, People Tracking module, and People Re-identifying module. These modules support a bunch of well-known people detectors, trackers, and re-identificators which can be used separately or put together with only a few line of codes. 
* **pyppbox** also supports real-time online and offline evaluation on [PoseTReID datasets](https://github.com/rathaumons/PoseTReID_DATASET).
* The initial version of pyppbox is also intergrated with GUI for easy demo and config. 

![alt text](https://raw.githubusercontent.com/rathaROG/screenshot/master/pyppbox/pyppbox_launchGUI.png)

## ⚽ Comparison Results

* Comparisions on PoseTReID datasets (More details? Check our paper: http://arxiv.org/abs/2205.10086)

<img src="https://raw.githubusercontent.com/rathaROG/screenshot/master/pyppbox/pyppbox_res001.png">

## ⚙️ Requirements

This is the very beginning of pyppbox, and we currently support only **Windows** due to the incompatibility of some modules on other platforms with the latest hardware.

### Install requirements
* Important: Only support **Python 3.9.x** and **CUDA 11.3.x** + **CUDNN 8.3.x**.
* Optional: You can create a virtual conda **Python 3.9** enviroment of your choice.
```
conda create --name pyppbox_env python=3.9
conda activate pyppbox_env
```
* Simply run the install_pippackages.cmd
```
cd requiremnts
install_pippackages.cmd
```

### Verify the requirements
* Simply run the testme.cmd
```
testme.cmd
```
* If there is no error, then you're all good and ready.

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

* Extension of PoseTReID paper:
```
@misc{https://doi.org/10.48550/arxiv.2205.10086,
  doi = {10.48550/ARXIV.2205.10086},
  url = {https://arxiv.org/abs/2205.10086},
  author = {Siv, Ratha and Mancas, Matei and Gosselin, Bernard and Valy, Dona and Sreng, Sokchenda},
  title = {People Tracking and Re-Identifying in Distributed Contexts: Extension of PoseTReID},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}}
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
