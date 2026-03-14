# compare_rl_baseline.py
import csv


BASELINE_SUMMARY = "results/fixed_time_summary.csv"
RL_SUMMARY = "results/rl_summary.csv"


def read_summary(path: str) -> dict:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise RuntimeError(f"No rows in summary file: {path}")
        return rows[0]


def main():
    baseline = read_summary(BASELINE_SUMMARY)
    rl = read_summary(RL_SUMMARY)

    def f(d, key):
        return float(d[key])

    print("=== Baseline vs RL Comparison ===")
    print(f"Baseline avg_vehicle_waiting_time: {f(baseline, 'avg_vehicle_waiting_time'):.3f}")
    print(f"RL       avg_vehicle_waiting_time: {f(rl, 'avg_vehicle_waiting_time'):.3f}")
    print()

    print(f"Baseline avg_vehicle_queue_length: {f(baseline, 'avg_vehicle_queue_length'):.3f}")
    print(f"RL       avg_vehicle_queue_length: {f(rl, 'avg_vehicle_queue_length'):.3f}")
    print()

    print(f"Baseline avg_ped_waiting_time: {f(baseline, 'avg_ped_waiting_time'):.3f}")
    print(f"RL       avg_ped_waiting_time: {f(rl, 'avg_ped_waiting_time'):.3f}")
    print()

    print(f"Baseline avg_ped_queue_length: {f(baseline, 'avg_ped_queue_length'):.3f}")
    print(f"RL       avg_ped_queue_length: {f(rl, 'avg_ped_queue_length'):.3f}")
    print()

    print(f"Baseline cumulative_reward: {f(baseline, 'cumulative_reward'):.1f}")
    print(f"RL       cumulative_reward: {f(rl, 'cumulative_reward'):.1f}")
    print()

    print(f"Baseline throughput: {f(baseline, 'throughput'):.0f}")
    print(f"RL       throughput: {f(rl, 'throughput'):.0f}")


if __name__ == "__main__":
    main()
