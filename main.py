from trainer.hierarchical_fl_trainer import HierarchicalFLTrainer

def main():
    num_devices = 10
    num_edges = 2

    trainer = HierarchicalFLTrainer(
        num_devices=num_devices,
        num_edges=num_edges
    )
    trainer.train()
    trainer.summarize()

if __name__ == "__main__":
    main()

