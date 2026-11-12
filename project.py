from dataset import *

NUM_SHOTS = 10_000_000

if __name__ == '__main__':
    circuit = make_circuit(p=9*10**-3, d=3, r=3)
    detection_events, observable_flips = make_dataset(circuit, num_shots=NUM_SHOTS)
    save_dataset(detection_events, observable_flips, 'data.npz')
    
    test_start = int(NUM_SHOTS * 0.9)
    test_obs = observable_flips[test_start:]
    test_events = detection_events[test_start:]
    n_test = len(test_obs)
    
    predictions, pymatching_time = pymatching_decoder(circuit, test_events)
    errors = count_logical_errors(test_obs, predictions)

    error_rate = errors / n_test
    uncertainty = (error_rate * (1 - error_rate) / n_test) ** 0.5

    print(f"Config: p = {9e-3}, d = 3, r = 3, test shots = {n_test:,}")
    print(f"Detectors per shot: {circuit.num_detectors}")
    print(f"Fraction of flips in test: {test_obs.mean():.4f}")
    print(f"PyMatching logical error rate: {error_rate:.4%} ± {uncertainty:.4%}")
    print(f"PyMatching time per shot: {pymatching_time / n_test * 1e6:.2f} µs")
