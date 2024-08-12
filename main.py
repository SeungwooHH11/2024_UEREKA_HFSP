# torch.autograd.set_detect_anomaly(True)
import torch
import vessl
torch.backends.cudnn.benchmark = True
from Simulator import *
from Network import *


def train_model(params, log_path=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    epoch = 0

    ave_act_loss = 0.0
    ave_cri_loss = 0.0
    ave_makespan = 0.0

    act_model = Act_net(params).to(device)
    cri_model = Cri_net(params).to(device)

    if params["optimizer"] == 'Adam':
        act_optim = optim.Adam(act_model.parameters(), lr=params["lr"])
        cri_optim = optim.Adam(cri_model.parameters(), lr=params["lr"])

    elif params["optimizer"] == "RMSProp":
        act_optim = optim.RMSprop(act_model.parameters(), lr=params["lr"])
        cri_optim = optim.RMSprop(cri_model.parameters(), lr=params["lr"])

    if params["is_lr_decay"]:
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])
        cri_lr_scheduler = optim.lr_scheduler.StepLR(cri_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])

    mse_loss = nn.MSELoss()

    t1 = time()
    for s in range(1, params["step"]):

        machines_per_stage = params['machines_per_stage']  # [1, 1, 1,1,1,1]
        processing_times = generate_processing_times(params["batch_size"], params["num_of_jobs"],
                                                     params['num_of_stages'], params['max_time'])
        pred_seq, ll_old, ps = act_model(processing_times.clone(), device)

        for k in range(params["k_epoch"]):
            real_makespan = calculate_makespans(params['num_of_stages'], machines_per_stage, processing_times, pred_seq).to(device).unsqueeze(-1)
            pred_makespan = cri_model(processing_times.clone(), device).unsqueeze(-1)


            adv = real_makespan.detach() - pred_makespan.detach()
            cri_loss = mse_loss(pred_makespan, real_makespan.detach())
            cri_optim.zero_grad()
            cri_loss.backward()
            nn.utils.clip_grad_norm_(cri_model.parameters(), max_norm=1., norm_type=2)
            cri_optim.step()
            if params["is_lr_decay"]:
                cri_lr_scheduler.step()
            ave_cri_loss += cri_loss.item()

            _, ll_new, _ = act_model(processing_times.clone(), device, pred_seq)
            ratio = torch.exp(ll_new - ll_old.detach()).unsqueeze(-1)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - params["epsilon"], 1 + params["epsilon"]) * adv
            act_loss = torch.max(surr1, surr2).mean()
            act_optim.zero_grad()
            act_loss.backward()
            act_optim.step()
            nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=1., norm_type=2)
            if params["is_lr_decay"]:
                act_lr_scheduler.step()
            ave_act_loss += act_loss.item()
        if s%100==0:
            vessl.log(step=s, payload={'ave_makespan':real_makespan.mean().item()})
        #ave_makespan += real_makespan.mean().item()


if __name__ == '__main__':

    load_model = True

    log_dir = "./result/log"
    if not os.path.exists(log_dir + "/ppo"):
        os.makedirs(log_dir + "/ppo")

    model_dir = "./result/model"
    if not os.path.exists(model_dir + "/ppo"):
        os.makedirs(model_dir + "/ppo")

    params = {
        "num_of_stages": 6,
        "num_of_jobs": 50,
        "max_time": 50,
        "n_embedding": 512,
        "n_hidden": 512,
        'machines_per_stage': [1, 1, 1, 1, 1, 1],
        "step": 20000,
        "log_step": 10,
        "log_dir": log_dir,
        "save_step": 1000,
        "model_dir": model_dir,
        "batch_size": 128,

        "init_min": -0.08,
        "init_max": 0.08,
        "use_logit_clipping": True,
        "C": 10,
        "T": 1.3,
        "decode_type": "sampling",
        "k_epoch": 2,
        "epsilon": 0.2,
        "optimizer": "Adam",
        "n_glimpse": 1,
        "n_process": 3,
        "lr": 0.001,
        "is_lr_decay": True,
        "lr_decay": 0.98,
        "lr_decay_step": 2000,
    }

    train_model(params)
