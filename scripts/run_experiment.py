from qoe_greenslicer_rl.experiment import run_experiment

if __name__ == "__main__":
    summary = run_experiment("results")
    print(summary.to_string(index=False))
