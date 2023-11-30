MLAGENTS_ENV_SPECIFIC_ARGS = {
    "3DBallHard": {'hidden_size': 128, 'max_steps': 20000},
    "Worm": {},
    "Crawler": {},
    "Walker": {'hidden_size': 256}, 
    "Hallway": {'hidden_size': 192, 'state_normalizer': 'none'},
    "PushBlock": {'hidden_size': 192, 'state_normalizer': 'none'},
    "Pyramids": {'state_normalizer': 'none', 'max_steps': 2000000}
}

GYM_ENV_SPECIFIC_ARGS = {
    "InvertedDoublePendulum-": {'reward_scale': 0.01, 'hidden_size': 128},   
    "Pusher-": {'reward_scale': 0.1, 'hidden_size': 160},   
    "Reacher-": {'reward_scale': 0.1, 'hidden_size': 160},
    "Hopper-": {'reward_scale': 0.1, 'hidden_size': 176},   
    "Walker2d-": {'reward_scale': 0.1, 'hidden_size': 176},   
    "Ant-": {'reward_scale': 0.1, 'hidden_size': 176},
    "HalfCheetah-": {'reward_scale': 0.1, 'hidden_size': 192},   
    "Humanoid-": {'reward_scale': 0.01, 'hidden_size': 256},    
    "HumanoidStandup-": {'reward_scale': 0.001, 'hidden_size': 256}
}
