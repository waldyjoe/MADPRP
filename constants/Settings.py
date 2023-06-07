# Parameters for preprocessing raw sector data
PRE_PROCESS_RAW_DATA = False
USE_REAL_TRAVEL_TIME = True
REDUCE_AGENT = 0.8 # Reduce the number of agents by 20%

# Input files that need to be read in
# For filename = './data/grid0.02_centroid_npc.geojson'
BASEMAP_FILE = "./data/RawData/grid0.02_npc.geojson"
TRAVEL_TIME_MATRIX = "./data/RawData/travel_time_matrix.pkl"
GRIDS_MEDOID = "./data/RawData/grids_medoid.pkl"
GRIDS_NODES_COUNT = "./data/RawData/grids_nodes_count.pkl"
ADJACENCY_MATRIX = "./data/RawData/adjacency_matrix.csv"

HQ_ID = 232  # 232
EXCLUSION_LIST = []

# OneMap Parameter
TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOjczNTIsInVzZXJfaWQiOjczNTIsImVtYWlsIjoid2FsZHkuam9" \
        "lLjIwMThAcGhkY3Muc211LmVkdS5zZyIsImZvcmV2ZXIiOmZhbHNlLCJpc3MiOiJodHRwOlwvXC9vbTIuZGZlLm9uZW1hcC5z" \
        "Z1wvYXBpXC92MlwvdXNlclwvc2Vzc2lvbiIsImlhdCI6MTYxNjU2NjA4NCwiZXhwIjoxNjE2OTk4MDg0LCJuYmYiOjE2MTY1N" \
        "jYwODQsImp0aSI6IjdmNWU4ZDQ2NTFhNGVkMWQ5YTc1YmE2MDMzMzVlNDQ3In0.IlEiayBoFIakYMHQ4YLX7rnzttyRO1ovFeeEQ9pHrOQ"

MIN_RESOLUTION_TIME = 10  # Minimum resolution time for an incident
GAMMA_A = 3  # Parameter for the Gamma distribution to represent the variability of incident resolution time

MAX_SERVICE_TIME = 70 # Max resolution time (in mins) per incident

RESOLUTION_TIME_DIST = "gamma"
# RESOLUTION_TIME_DIST = "uniform"

# Parameters for operational requirements
TAU_TARGET = 20  # The response time needed to attend to an incident in mins
SHIFT_DURATION = 720  # In minutes
BETA_R = 25  # For response utility function
BETA_P = 2.5  # For presence utility function

# Parameters for simulation environment
START_SHIFT = 8  # Time when the shift starts
TIME_UNIT = 10  # No. of minutes that one unit time in the simulation environment represents
RESPONSE_PERIOD = 3  # No. of time unit where incident can be attended by the available patrol agent
T = [n for n in range(0, SHIFT_DURATION + 1) if n % TIME_UNIT == 0]  # Time horizon

# Rescheduling Policy
MAX_PERTURBATION_DISTANCE = 0.4  # 20%
TIME_LIMIT_H = 1  # in seconds
EPSILON_H = 0.3

# Parameters for Best Response Procedure
MAX_ITER = 100
EPSILON_B = 0.7
TIME_LIMIT_B = 1  # in seconds

# Parameter for Local Value Network
ACTION_SIZE = 1
FC1_UNITS = 64
FC2_UNITS = 32
HIDDEN_DIM = 64
ATTN_OUT_DIM = 32

# Parameters for learning VFA
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size 20
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # 5e-3 #5e-4              # learning rate
UPDATE_EVERY = 20  # how often to update the network 10
LEARN_EVERY = 10  # 10
SEED = 1
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2500

# Params for initial schedule solver (dispatch rules)
MAX_ITER_SOLVER = 3
PATROL_WINDOW = 2
SA_TEMP = 0.1
SA_TEMP_DECAY_RATE = 0.5

# Parameters for parallel runs
CPU_UTIL = 0.6
