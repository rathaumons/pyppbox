"""
    pyppbox: Toolbox for people detecting, tracking, and re-identifying.
    Copyright (C) 2022 UMONS-Numediart

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import os
import os.path
import skbuild


def main():

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    package_version = get_and_set_info_static()

    package_name = "pyppbox"

    long_description = "Pure Python toolbox for people detecting, tracking, and re-identifying... "

    packages = ["pyppbox", "pyppbox.cfg", "pyppbox.dt_yolocv", "pyppbox.gui", "pyppbox.ri_deepreid", 
                "pyppbox.ri_deepreid.classifier", "pyppbox.ri_deepreid.classifier.gta5p5", 
                "pyppbox.ri_deepreid.data", "pyppbox.ri_deepreid.pretrained", "pyppbox.ri_deepreid.pretrained.base", 
                "pyppbox.ri_deepreid.pretrained.torchreid", "pyppbox.ri_facenet", "pyppbox.ri_facenet.classifier", 
                "pyppbox.ri_facenet.classifier.gta5p5", "pyppbox.ri_facenet.data", "pyppbox.ri_facenet.models", 
                "pyppbox.ri_facenet.models.20180402-114759", "pyppbox.ri_facenet.models.det", "pyppbox.tk_centroid", 
                "pyppbox.tk_deepsort", "pyppbox.tk_sort", "pyppbox.tmp", "pyppbox.tmp.demo", "pyppbox.tmp.gt", 
                "pyppbox.tmp.res", "pyppbox.utils"]


    package_data = {
        "pyppbox": ["*"], 
        "pyppbox.cfg": ["*"], 
        "pyppbox.dt_yolocv": ["*"], 
        "pyppbox.gui": ["*"], 
        "pyppbox.ri_deepreid": ["*"], 
        "pyppbox.ri_deepreid.classifier": ["*"], 
        "pyppbox.ri_deepreid.classifier.gta5p5": ["*"], 
        "pyppbox.ri_deepreid.data": ["*"], 
        "pyppbox.ri_deepreid.pretrained": ["*"], 
        "pyppbox.ri_deepreid.pretrained.base": ["*"], 
        "pyppbox.ri_deepreid.pretrained.torchreid": ["*"], 
        "pyppbox.ri_facenet": ["*"], 
        "pyppbox.ri_facenet.classifier": ["*"], 
        "pyppbox.ri_facenet.classifier.gta5p5": ["*"], 
        "pyppbox.ri_facenet.data": ["*"], 
        "pyppbox.ri_facenet.models": ["*"], 
        "pyppbox.ri_facenet.models.20180402-114759": ["*"], 
        "pyppbox.ri_facenet.models.det": ["*"], 
        "pyppbox.tk_centroid": ["*"], 
        "pyppbox.tk_deepsort": ["*"], 
        "pyppbox.tk_sort": ["*"], 
        "pyppbox.tmp": ["*"], 
        "pyppbox.tmp.demo": ["*"], 
        "pyppbox.tmp.gt": ["*"], 
        "pyppbox.tmp.res": ["*"], 
        "pyppbox.utils": ["*"], 
    }

    skbuild.setup(
        name=package_name,
        version=package_version,
        url="https://github.com/rathaumons/pyppbox",
        license="GPLv3+",
        description="Toolbox for people detecting, tracking, and re-identifying",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=packages,
        package_data=package_data,
        maintainer="rathaROG",
        ext_modules=EmptyListWithLength(),
        install_requires=None,
        python_requires=">=3.9",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Information Technology",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
            "Operating System :: Microsoft :: Windows",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Image Recognition",
            "Topic :: Software Development",
        ],

    )


def get_and_set_info_static():
    return "1.0b12"


class EmptyListWithLength(list):
    def __len__(self):
        return 1


if __name__ == "__main__":
    main()
