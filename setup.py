# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2022 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


from setuptools import find_packages, setup


def read_requirements():
    with open("requirements/requirements.txt") as req_txt:
        return [line for line in req_txt.read().splitlines()]

def get_version_string():
    version_py = "pyppbox/__init__.py"
    with open(version_py) as version_file:
        for line in version_file.read().splitlines():
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]

def get_packages():
    import yaml
    from yaml.loader import SafeLoader
    packages = find_packages()
    package_data = {}
    with open("setup_extra.yaml") as setup_extra:
        extra_dict = yaml.load(setup_extra, Loader=SafeLoader)
        packages = packages + extra_dict['extra_data']
        packages = list(set(packages))
        for p in packages: package_data.update({p: ["*"]})
    return packages, package_data

def main_setup():
    long_description = open("README.md", encoding="utf-8").read()
    packages, package_data = get_packages()
    setup(
        name="pyppbox",
        version=get_version_string(),
        url="https://github.com/rathaumons/pyppbox",
        license="GPL-3.0-or-later",
        description="Toolbox for people detecting, tracking, and re-identifying.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=packages,
        package_data=package_data,
        include_package_data=True,
        author="Ratha SIV",
        maintainer="rathaROG",
        install_requires=read_requirements(),
        keywords=['Toolbox', 'People Detecting', 'People Tracking', 'People Re-Identification'],
        python_requires=">=3.9",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Information Technology",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Image Recognition",
            "Topic :: Software Development",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
    )

if __name__ == "__main__":
    main_setup()
