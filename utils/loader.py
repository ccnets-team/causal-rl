import os
import torch.utils.data

def load_trainer(trainer):
    opimizers = trainer.get_optimizers()
    schedulers = trainer.get_schedulers()
    networks = trainer.get_networks()
    target_networks = trainer.get_target_networks()
    state_normalizer = trainer.get_state_normalizer() 
    reward_normalizer = trainer.get_reward_normalizer()
    
    for idx, it in enumerate(target_networks):
        if it is not None:
            it.load_state_dict(torch.load(trainer.save_path + f"target_network_{idx}"+ '.pth', map_location="cuda:0"))
    
    if state_normalizer is not None:
        state_normalizer.load(trainer.save_path + f"state_normalizer" + '.pth')
    
    if reward_normalizer is not None:
        reward_normalizer.load(trainer.save_path + f"reward_normalizer" + '.pth')

    for idx, it in enumerate(networks):
        it.load_state_dict(torch.load(trainer.save_path + f"network_{idx}"+ '.pth', map_location="cuda:0"))
    
    for idx, it in enumerate(opimizers):
        it.load_state_dict(torch.load(trainer.save_path + f"opt_{idx}" + '.pth', map_location="cuda:0"))
        it.param_groups[0]['capturable'] = True
        
    for idx, it in enumerate(schedulers):
        it.load_state_dict(torch.load(trainer.save_path + f"sch_{idx}" + '.pth', map_location="cuda:0"))
        
def save_trainer(trainer, save_path):
    opimizers = trainer.get_optimizers()
    schedulers = trainer.get_schedulers()
    networks = trainer.get_networks()
    target_networks = trainer.get_target_networks()
    state_normalizer = trainer.get_state_normalizer() 
    reward_normalizer = trainer.get_reward_normalizer()

    for idx, it in enumerate(networks):
        torch.save(it.state_dict(), os.path.join(save_path, f"network_{idx}" + '.pth'))

    for idx, it in enumerate(opimizers):
        torch.save(it.state_dict(), os.path.join(save_path, f"opt_{idx}" + '.pth'))
    for idx, it in enumerate(schedulers):
        torch.save(it.state_dict(), os.path.join(save_path, f"sch_{idx}" + '.pth'))

    for idx, it in enumerate(target_networks):
        if it is not None:
            torch.save(it.state_dict(), os.path.join(save_path, f"target_network_{idx}" + '.pth'))

    if state_normalizer is not None:
        state_normalizer.save(os.path.join(save_path, f"state_normalizer" + '.pth'))

    if reward_normalizer is not None:
        reward_normalizer.save(os.path.join(save_path, f"reward_normalizer" + '.pth'))
