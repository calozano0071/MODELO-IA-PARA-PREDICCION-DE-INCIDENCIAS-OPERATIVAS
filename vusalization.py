import matplotlib.pyplot as plt


def plot_loss(history, out_path=None):
    plt.figure()
    plt.plot(history.history.get('loss', []), label='loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend()
    plt.title('Loss')
    if out_path:
        plt.savefig(out_path)
    plt.close()


def plot_auc(history, out_path=None):
    plt.figure()
    plt.plot(history.history.get('auc', []), label='auc')
    plt.plot(history.history.get('val_auc', []), label='val_auc')
    plt.legend()
    plt.title('AUC')
    if out_path:
        plt.savefig(out_path)
    plt.close()
