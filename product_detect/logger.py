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

def wnb_write(like_json):
    wandb.log(like_json)

def wnb_close():
    wandb.finish()

def wnb_write_conf_mat(labels, preds, num_class):
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(
                probs=None,
                y_true=labels, 
                preds=preds, 
                class_names=range(num_class))}) 
