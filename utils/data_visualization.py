import matplotlib.pyplot as plt


def draw_training_graph(objective_func_vals):
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()


def draw_data_prediction(x, y_pred, y_true):
    for x, y_target, y_p in zip(x, y_true, y_pred):
        if y_target == 1:
            plt.plot(x[0], x[1], 'bo')
        else:
            plt.plot(x[0], x[1], 'go')
        if y_target != y_p:
            plt.scatter(x[0], x[1], s=200, facecolors='none', edgecolors='r', linewidths=2)
    plt.show()

def circuit_draw(qc, style = "mpl"):
    return qc.draw(style, 
                    style={'subfontsize':7, 
                            'fontsize':20, 
                            'compress':False}, 
                    vertical_compression='high')
