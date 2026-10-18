# 🐍📦 [**pyppbox**](https://github.com/rathaumons/pyppbox)

<div align="center">

[![Documentation](https://github.com/rathaumons/pyppbox/actions/workflows/pyppboxdocs.yaml/badge.svg)](https://github.com/rathaumons/pyppbox/actions/workflows/pyppboxdocs.yaml) [![Build PyPI](https://github.com/rathaumons/pyppbox/actions/workflows/build_pypi.yaml/badge.svg)](https://github.com/rathaumons/pyppbox/actions/workflows/build_pypi.yaml) [![Test Build](https://github.com/rathaumons/pyppbox/actions/workflows/test_build.yaml/badge.svg)](https://github.com/rathaumons/pyppbox/actions/workflows/test_build.yaml)

[![Test Core Windows](https://github.com/rathaumons/pyppbox/actions/workflows/test_core_windows.yaml/badge.svg)](https://github.com/rathaumons/pyppbox/actions/workflows/test_core_windows.yaml) [![Test Core Linux](https://github.com/rathaumons/pyppbox/actions/workflows/test_core_linux.yaml/badge.svg)](https://github.com/rathaumons/pyppbox/actions/workflows/test_core_linux.yaml) [![Test Core macOS](https://github.com/rathaumons/pyppbox/actions/workflows/test_core_macos.yaml/badge.svg)](https://github.com/rathaumons/pyppbox/actions/workflows/test_core_macos.yaml)

<img src="https://raw.githubusercontent.com/rathaROG/screenshot/master/pyppbox/pyppbox_new_wide.png"><br />

**[📗 Documentation](https://rathaumons.github.io/pyppbox/) | [🚀 Getting started](https://rathaumons.github.io/pyppbox/getstarted.html) | [💡 Examples](https://rathaumons.github.io/pyppbox/examples.html) | [▶️ More demo](https://github.com/rathaumons/pyppbox-demo)**

</div>

***pyppbox*** is an all-in-one Python toolbox which can be used for detecting, tracking, and re-identifying people with only a few lines of code. It is originally made for [***PoseTReID framework***](https://github.com/rathaumons/PoseTReID_DATASET) which can effectively track specific/certain people across multiple cameras or video scenes in distributed contexts of people interaction spaces such as malls or amusement parks, etc.

* ` pyppbox = Python + People + Toolbox `
* Design for both short and long term people detecting, tracking, and re-identifying.
* Integrated GUI for configuration and demos, with editable device choices and automatic-device rows. See [GUI and device settings](https://rathaumons.github.io/pyppbox/pyppbox/config.html#device-selection).
* Support dictionary and YAML/JSON configuration.
* Optional ArcFace and AdaFace face recognition with ONNX Runtime, SCRFD alignment, and model-specific identity classifiers. See [face recognizer setup](https://rathaumons.github.io/pyppbox/pyppbox/modules/reiders.html).
* Use the eight [v1.6.0 model data wheels](https://github.com/rathaumons/pyppbox-data/releases/tag/v1.6.0), including the retrained GTA V identity classifiers. See [installation and compatibility](https://rathaumons.github.io/pyppbox/getstarted.html).
* Support [[GTA_V_Dataset]](https://github.com/rathaumons/PoseTReID_DATASET) -> Real-time online and offline evaluation.
* Check our papers: 1 [[IEEE]](https://ieeexplore.ieee.org/document/9271712) | 2 [[IEEE]](https://ieeexplore.ieee.org/document/9946587) [[arxiv]](https://doi.org/10.48550/arxiv.2205.10086)
