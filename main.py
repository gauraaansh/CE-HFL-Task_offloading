from trainer.independent_trainer import IndependentTrainer
from trainer.flat_fl_trainer import FlatFLTrainer
from trainer.hierarchical_fl_trainer import HierarchicalFLTrainer
from utils.plot_results import plot_curves

def main():

    results = {}

#    ind = IndependentTrainer(10)
#    ind.train()
#    results["Independent"] = (ind.reward_history, [0]*len(ind.reward_history))

    flat = FlatFLTrainer(10)
    flat.train()
    results["Flat FL"] = (flat.reward_history, flat.comm_history)

    hier = HierarchicalFLTrainer(10, 2)
    hier.train()
    results["Hierarchical FL"] = (hier.reward_history, hier.comm_history)

    plot_curves(results)

if __name__ == "__main__":
    main()
