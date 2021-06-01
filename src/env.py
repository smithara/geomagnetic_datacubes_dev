"""
"""


from os.path import dirname, abspath, join
from configparser import ConfigParser

REFRAD = 6371200

# Path to the project root
PROJ_DIR = dirname(
    dirname(
        abspath(__file__)
    )
)

# Use the configuration file in the project root
config = ConfigParser()
config.read(join(PROJ_DIR, "config.ini"))

###############################################################################
# Data directory locations
TMPDIR = config["Temporary Directory"].get("TMPDIR", None)
DATA_RAW_DIR = join(PROJ_DIR, "data/raw/")
DATA_INT_DIR = join(PROJ_DIR, "data/interim/")
DATA_EXT_DIR = join(PROJ_DIR, "data/external")
DATA_PROCD_DIR = join(PROJ_DIR, "data/processed")
MODEL_DIR = join(PROJ_DIR, "src/models")

# ###############################################################################
# # Figures directory
# FIGDIR = join(PROJ_DIR, "reports/figures")

# ###############################################################################
# # Auxiliary file locations
# # File from BGS
# BADDATALIST = join(DATA_EXT_DIR, "badDataPatterns.lst")
# # RC index from http://www.spacecenter.dk/files/magnetic-models/RC/
# RC_FILE = join(DATA_EXT_DIR, "RC_1997-2019_augmented.dat")
# # Icosphere file
ICOS_FILE = join(DATA_EXT_DIR, "icosphere_data.h5")
# # AEJ estimate file
# AEJ_FILES = {
#     sat_ID: join(DATA_PROCD_DIR, f"AEJ_Sw{sat_ID}.h5")
#     for sat_ID in "ABC"
# }

# ##############################################################################
# # Generated models
# MOD_2H_AB = join(MODEL_DIR, "mod2h_AB.shc")
# MOD_3H_AB = join(MODEL_DIR, "mod3h_AB.shc")
# # External models
# MOD_LCS = join(DATA_EXT_DIR, "LCS-1.shc")
# MOD_MF7 = join(DATA_EXT_DIR, "MF7.cof")
# MOD_CHAOS_CRUST = join(DATA_EXT_DIR, "CHAOS-6_static_full.shc")
# MOD_EMAG2V3 = join(DATA_EXT_DIR, "EMAG2V3/EMAG2_V3_20170530.csv")
# MOD_SWARM_MLI_2D_0501 = join(DATA_EXT_DIR, "SW_OPER_MLI_SHA_2D_00000000T000000_99999999T999999_0501.shc")
