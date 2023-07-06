import wandb
from sklearn.metrics import confusion_matrix

def wnb_page_init(epochs, project, name, wnb_api_key):
    wandb.login(key=wnb_api_key)
    wandb.init(
        project=project,
        name=name,
        config={
        # "architecture": "efficientNet",
        "epochs": epochs,
        }
    )

def wnb_write(like_json, epoch):
    wandb.log(like_json, step=epoch)

def wnb_close():
    wandb.finish()

def wnb_write_conf_mat(epoch, labels, preds, num_class):
    wandb.log({f"conf_mat{epoch}" : wandb.plot.confusion_matrix(
                y_true=labels, 
                preds=preds, 
                class_names=range(num_class))})
    
def wnb_watch(model, criterion):
    wandb.watch(model, criterion, log="all", log_freq=1)
