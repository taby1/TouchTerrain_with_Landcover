from touchterrain.common import config
import os

# TouchTerrain server config settings

# Location of the file containing the google maps key
GOOGLE_MAPS_KEY_FILE = os.getenv('TOUCHTERRAIN_GOOGLE_MAPS_KEY_FILE', os.path.join(config.SERVER_DIR, 'GoogleMapsKey.txt'))


#
# 5/2025: changed recaptcha file location to /tmp to make them not web accessible. 
# Miles made /tmp read-only so it doesn't get wiped on rebuilds
#

# file for Recptcha v3 keys
#RECAPTCHA_V3_KEYS_FILE = os.path.join(config.SERVER_DIR, 'Recaptcha_v3_keys.txt')
RECAPTCHA_V3_KEYS_FILE = "/tmp/Recaptcha_v3_keys.txt"

# log for recaptcha v3
RECAPTCHA_V3_LOG_FILE = os.path.join(config.SERVER_DIR, 'Recaptcha_v3_log.txt')


# DEBUG_MODE will be True if running in a local development environment.
DEBUG_MODE = ('SERVER_SOFTWARE' in os.environ and
              os.environ['SERVER_SOFTWARE'].startswith('Dev'))

# Defaults

# type of server:
#SERVER_TYPE = "flask_local" # so I can run the server inside a debugger, needs to run with single core!
SERVER_TYPE = "gnunicorn"

# multiprocessing: This will not work under gnunicorn but does work on my local Win10 dev server
# It should also work whe using standalone mode.
NUM_CORES = 1 # 0 means: use all cores, 1 means: use one core, etc. None mean 1 core
if SERVER_TYPE == "flask_local": NUM_CORES = 1 # 1 means don't use multi-core at all


# limits for ISU server

# for STL/OBJ don't even start with a DEM bigger than that number. GeoTiff export is this * 100!
#MAX_CELLS_PERMITED =   1000 * 1000 * 4  # private
MAX_CELLS_PERMITED =   1000 * 1000 * 0.7

# if DEM has > this number of cells, use tempfile instead of memory
MAX_CELLS = MAX_CELLS_PERMITED / 4

# folders
TMP_FOLDER = os.getenv('TOUCHTERRAIN_TMP_FOLDER', os.path.join(config.SERVER_DIR, "tmp"))
DOWNLOADS_FOLDER = os.getenv('TOUCHTERRAIN_DOWNLOADS_FOLDER', os.path.join(config.SERVER_DIR, "downloads"))
PREVIEWS_FOLDER = os.getenv('TOUCHTERRAIN_PREVIEWS_FOLDER', os.path.join(config.SERVER_DIR, "previews"))

# This will be inlined in index.html to enable Google Analytics, However, this is
# my tracking id, so if you use google analytics, make sure to use your own Tracking ID!
GOOGLE_ANALYTICS_TRACKING_ID = "G-EGX5Y3PBYH"
# If you don't wan to use GA, set this to "" !
