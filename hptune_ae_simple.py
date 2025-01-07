from functools import partial
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import ray
from ray import tune, air
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bohb import TuneBOHB

from data.new import build_ens_allvar_dataset
from model.ae_simple import AEens
from utils import EarlyStopper, device


ray.init(num_cpus=38, num_gpus=1)

VAR = 'v10'
BATCH_SIZE = 1024
N_LATENT = 2

def train_ae(config):
    net = AEens(config["latent_dim"], config["n_nodes1"], config["n_nodes2"], config["n_layers"], config["activation"])
    net.to(device)
                 
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
#    early_stopper = EarlyStopper(patience=3, min_delta=0)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        net.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    train, val, test, _ = build_ens_allvar_dataset(var=VAR)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE)

    for epoch in range(start_epoch, 20):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        #net.train()
        
        for batch, data in enumerate(train_loader, 0):
            inputs = data[0].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs.float())
            loss_all = net.loss_function(*outputs)
            loss = loss_all.get('loss')
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if batch % 200 == 0:  # print every 200 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, batch + 1, running_loss / epoch_steps)
                )
                print(f"RMSE: {loss_all.get('RMSE'):>3f}.")
    
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        rmse_loss = 0.0
        #net.eval()
        for batch, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs = data[0].to(device)
                outputs = net(inputs.float())
                loss_all = net.loss_function(*outputs)
                loss = loss_all.get('loss')
                #val_loss += loss.cpu().numpy()
                val_loss += loss.item()
                val_steps += 1
                rmse = loss_all.get('RMSE')
                rmse_loss += rmse.item()

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / val_steps}#, "accuracy": rmse_loss / val_steps}#,
            #checkpoint=checkpoint
        )
        
#        if early_stopper.early_stop(val_loss/val_steps):
#            print("Early stopping at epoch:", t)
#            break
            
    print("Finished Training")
    
    
def test_accuracy(net):
    train, val, test, _ = build_ens_allvar_dataset(var=VAR)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE)

    test_loss = 0.0
    test_steps = 0
    test_rmse = 0.0
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images = data[0].to(device)
            outputs = net(images.float())
            loss_all = net.loss_function(*outputs)
            loss = loss_all.get('loss')
            test_loss += loss.item()
            test_steps += 1
            #rmse = loss_all.get('RMSE')
            #test_rmse += rmse.item()
            
    return test_loss / test_steps



def main(num_samples=50, max_num_epochs=10, gpus_per_trial=1):

    train, val, test, _ = build_ens_allvar_dataset(var=VAR)
    config_space = {
        "latent_dim": tune.choice([N_LATENT]),
        "n_nodes1": tune.choice([128, 256, 512, 1024, 2048, 4096]),
        "n_nodes2": tune.choice([128, 256, 512, 1024, 2048, 4096]),
        "n_layers": tune.choice([1, 2, 3]),
        "activation": tune.choice(['none','gelu','leakyrelu','relu'])
    }

    algo_bos = BayesOptSearch(metric="loss", mode="min")
    algo_bohb = TuneBOHB(metric="loss", mode="min")
    algo_optuna = OptunaSearch(metric="loss", mode="min")

    sche_bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        reduction_factor=4.0
    )
    sche_asha = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_ae),
            resources={"cpu": 38, "gpu": gpus_per_trial}
        ),
        param_space=config_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=sche_bohb,
            search_alg=algo_bohb
        ),
        run_config=air.RunConfig(
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=False,
                checkpoint_frequency=0,
                num_to_keep=1,
            )
            #local_dir=log_dir
        )
    )
    
    result = tuner.fit()
    
#    result = tune.run(
#        partial(train_ae),
#        resources_per_trial={"cpu": 38, "gpu": gpus_per_trial},
#        config=config,
#        num_samples=num_samples,
#        scheduler=sche_bohb,
#        search_alg=algo_bohb,
#        checkpoint_config=air.CheckpointConfig(
#            checkpoint_at_end=False,
#            checkpoint_frequency=0,
#            num_to_keep=1
#        )
#    )

#    best_trial = result.get_best_trial("accuracy", "min", "last")
    
    best_trial = result.get_best_result("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    #print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    #print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = AEens(best_trial.config["latent_dim"], best_trial.config["n_nodes1"], best_trial.config["n_nodes2"], best_trial.config["n_layers"], best_trial.config["activation"])
    best_trained_model.to(device)

#    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
#    best_checkpoint_data = best_checkpoint.to_dict()
#
#    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    test_acc = test_accuracy(best_trained_model)
    print("Best trial test set loss: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=60, max_num_epochs=20, gpus_per_trial=1)
    #main(num_samples=4, max_num_epochs=2, gpus_per_trial=1)




