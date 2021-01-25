from utils import BipedalWalkerWrapper

class train_params:
    ENV = 'BipedalWalkerHardcore-v3'
    RENDER = False
    RANDOM_SEED = 99999999
    NUM_AGENTS = 4
    dummy_env = BipedalWalkerWrapper()
    STATE_DIMS = dummy_env.get_state_dims()
    STATE_BOUND_LOW, STATE_BOUND_HIGH = dummy_env.get_state_bounds()
    ACTION_DIMS = dummy_env.get_action_dims()
    ACTION_BOUND_LOW, ACTION_BOUND_HIGH = dummy_env.get_action_bounds()
    V_MIN = dummy_env.v_min
    V_MAX = dummy_env.v_max
    del dummy_env
    BATCH_SIZE = 256
    NUM_STEPS_TRAIN = 1000
    MAX_EP_LENGTH = 10000
    REPLAY_MEM_SIZE = 1000000
    REPLAY_MEM_REMOVE_STEP = 200
    PRIORITY_ALPHA = 0.6
    PRIORITY_BETA_START = 0.4
    PRIORITY_BETA_END = 1.0
    PRIORITY_EPSILON = 0.00001
    NOISE_SCALE = 0.3
    NOISE_DECAY = 0.9999
    DISCOUNT_RATE = 0.99
    N_STEP_RETURNS = 5
    UPDATE_AGENT_EP = 10
    CRITIC_LEARNING_RATE = 0.0001
    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_L2_LAMBDA = 0.0
    DENSE1_SIZE = 400
    DENSE2_SIZE = 300
    FINAL_LAYER_INIT = 0.003
    NUM_ATOMS = 51
    TAU = 0.001
    USE_BATCH_NORM = False
    SAVE_CKPT_STEP = 10000
    CKPT_DIR = './saved_models/' + ENV
    CKPT_FILE = None
    LOG_DIR = None
    
    
class test_params:
    ENV = train_params.ENV
    RENDER = False
    RANDOM_SEED = 999999
    NUM_EPS_TEST = 100
    MAX_EP_LENGTH = 10000
    CKPT_DIR = './saved_models/' + ENV
    CKPT_FILE = 'ishan'
    RESULTS_DIR = './test_results'
    LOG_DIR = None

    

