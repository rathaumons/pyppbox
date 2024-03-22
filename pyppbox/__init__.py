"""

A toolbox for people detecting, tracking, and re-identifying

    * * * * * * * * * * * * * * * * * * * * * * * *
    *  Copyright (C) 2022 UMONS-Numediart GPLV3+  *
    * * * * * * * * * * * * * * * * * * * * * * * *

>>> pyppbox.launchGUI() # Launch GUI
>>> pyppbox.docs() # Read online documentation
>>> pyppbox.github() # Go to our GitHub

"""

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


from pyppbox.utils.commontools import github, docs
from pyppbox.gui.guitools import (
    launchGUI, 
    showMainConfig, 
    showAllDTConfig, 
    showAllRIConfig, 
    showAllTKConfig, 
    useInternalConfigDir, 
    useThisConfigDir,
    resetInternalConfig,
    generateConfig
)


__version__ = '3.5b1'
__author__ = 'Ratha SIV'
__description__ = 'Toolbox for people detecting, tracking, and re-identifying.'
__homepage__ = 'https://rathaumons.github.io/pyppbox'
__url__ = 'https://github.com/rathaumons/pyppbox.git'

