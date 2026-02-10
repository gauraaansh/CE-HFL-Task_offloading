import matplotlib.pyplot as plt

def plot_curves(results):
    # results = dict {name: (rewards, comm)}

    plt.figure()
    for name, (rewards, _) in results.items():
        plt.plot(rewards, label=name)

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig("reward_curve.png")
    plt.close()


    plt.figure()
    for name, (_, comm) in results.items():
        plt.plot(comm, label=name)

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Communication (bytes)")
    plt.legend()
    plt.savefig("communication_curve.png")
    plt.close()
