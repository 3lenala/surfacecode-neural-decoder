from dataset import *

NUM_SHOTS = 10_000_000

DECODERS = {"PyMatching": pymatching_decoder}

def evaluate(circuit: stim.Circuit, test_start: int, filename: str, decoder: str):
    detection_events, observable_flips = load_dataset(filename)
    
    test_obs = observable_flips[test_start:]
    test_events = detection_events[test_start:]
    n_test = len(test_obs)
    
    predictions, time = DECODERS[decoder](circuit, test_events)
    errors = count_logical_errors(test_obs, predictions)

    error_rate = errors / n_test
    uncertainty = (error_rate * (1 - error_rate) / n_test) ** 0.5

    return print_eval(n_test, error_rate, uncertainty, test_obs.mean(), time, decoder)
 
def print_eval(n_test: int, error_rate: float, uncertainty: float, flip_rate: float, pym_time: float, decoder: str) -> None:
    print(f"Config: p = {9e-3}, d = 3, r = 3, test shots = {n_test:,}")
    print(f"Detectors per shot: {circuit.num_detectors}")
    print(f"Fraction of flips in test: {flip_rate:.4f}")
    print(f"{decoder} logical error rate: {error_rate:.4%} ± {uncertainty:.4%}")
    print(f"{decoder} time per shot: {pym_time / n_test * 1e6:.2f} µs")


if __name__ == '__main__':
    circuit = make_circuit(p=9*10**-3, d=3, r=3)
    detection_events, observable_flips = make_dataset(circuit, num_shots=NUM_SHOTS)
    save_dataset(detection_events, observable_flips, 'data.npz')
    
    test_start = int(NUM_SHOTS * 0.9)
    evaluate(circuit, test_start, 'data.npz', "PyMatching")
    

