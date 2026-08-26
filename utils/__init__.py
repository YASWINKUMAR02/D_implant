# AI Dental Implant Planning System - Utilities
import os
import sys
import site

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

roaming_site = os.path.expandvars(r"%APPDATA%\Python\Python311\site-packages")
if os.path.exists(roaming_site) and roaming_site not in sys.path:
    sys.path.insert(0, roaming_site)
