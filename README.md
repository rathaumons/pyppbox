# Welcome to pyppbox !

## 🐍📦 pyppbox = Python  + People + Toolbox 

* Yes, pyppbox is an open-source Python toolbox which was originally made for the PoseTReID framework. 
* This toolbox features 3 main modules: People Detection module, People Tracking module, and People Re-identifying module. These modules support a bunch of well-known people detectors, trackers, and re-identificators which can be used separately or put together with only a few lines of code. 
* **pyppbox** also supports real-time online and offline evaluation on [PoseTReID datasets](https://github.com/rathaumons/PoseTReID_DATASET).
* The initial version of pyppbox is also intergrated with GUI for easy demo and config. 

<!--- ![alt text](https://raw.githubusercontent.com/rathaROG/screenshot/master/pyppbox/pyppbox_launchGUI.png) --->

## ⚠️ IMPORTANT NOTICES (July 27, 2022)

* Add GPLv3+ license
* OpenPose is completely removed due to [the complicated license of OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE) and it will be replaced by [MMPose](https://github.com/open-mmlab/mmpose) in the future versions. Stay tune for this amazing update!
* Please refer to [this repo pyppbox-paper](https://github.com/rathaumons/pyppbox-paper) for those who want OpenPose or want to reproduce the results in the paper.

## ⚽ Comparison Results

* Comparisions on PoseTReID datasets (Check our pre-printed paper: http://arxiv.org/abs/2205.10086)
* [Click here for raw results in the paper](https://drive.google.com/open?id=13pVqKKd0mtoAaVQh1USxOwZwxg4HmzyQ)

<img src="https://raw.githubusercontent.com/rathaROG/screenshot/master/pyppbox/pyppbox_res001n.png">

## ⚙️ Requirements

* Please check the [README.md](requirements/README.md) in [requirements](requirements) (Can be installed later).

## 🚀 Setup pyppbox

### Option 1: Use the [prebuilt WHL file here](https://drive.google.com/open?id=1LY5WNsSoEMwxYjET26yry4AQMHiPDaM0) 
* `pip install pyppbox-xxx.whl`

### Option 2: Build your own pyppbox
* Download [the extra models & weights](https://drive.google.com/open?id=1EQzkwZ8aCpZqGrgxEo6PD_8hPhHlcaqz) and extract to the root `pyppbox`
* Create WHL by running `creat_whl.cmd`
* Install newly created WHL `pip install pyppbox-xxx.whl`

### Quick Test
* On your terminal or CMD:
```
python
import pyppbox
pyppbox.launchGUI()
```
* Now you should see [the GUI of pyppbox for easy demo](https://raw.githubusercontent.com/rathaROG/screenshot/master/pyppbox/pyppbox_launchGUI_main.png). Just hit the "LAUNCH" button!

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
