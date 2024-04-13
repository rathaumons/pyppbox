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


from pyppbox.ppb import *
from pyppbox.utils.logtools import add_warning_log

msg = ("\n! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !\n"
       "!                                                                       !\n"
       "!  For the coming major version 4:                                      !\n"
       "!   * 'pyppbox' will leverage 'vsensebox' for detection and tracking    !\n"
       "!   * 'pyppbox.utils.persontools' will change to adapt 'vesensebox'     !\n"
       "!   * 'pyppbox.ppb' will replace 'pyppbox.standalone'                   !\n"
       "!   * All configuration files will also change                          !\n"
       "!   * All changes will added to release notes                           !\n"
       "!                                                                       !\n"
       "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !\n")

add_warning_log(msg)

