import time
from copy import deepcopy
import mlrose_hiive
import matplotlib.pyplot as plot
import numpy as np  

def param_tune_ga(algo, reset=2):    # reset indicates trying 2 times
    pop_sizes = range(100, 501, 100)
    mutation_prob = [0.005, 0.05, 0.1, 0.5]
    fitness_values = [[] for _ in range(len(mutation_prob))]
    for i, mu in enumerate(mutation_prob):
        for sz in pop_sizes:
            fits = []
            times = []
            for _ in range(reset):
                start = time.time()
                _, fitness_value, _ = mlrose_hiive.genetic_alg(algo, pop_size=sz, mutation_prob=mu)
                fits.append(fitness_value)
                times.append(time.time() - start)
                fits.append(fitness_value)
            fitness_values[i].append(np.mean(fits))
    # Plot Curve for optimal value
    param_opt_graph(np.array(fitness_values),f'Optimizing Genetic algorithm - {repr(algo.fitness_fn).split(".")[-1].split(" ")[0]}','Population_Size','Mutation_Rate',
        pop_sizes,mutation_prob)

def param_tune_sa(algo, reset=2):
    init_temps = np.linspace(1.0, 10.0, 10)
    decay_rates = np.linspace(0.1, 0.99, 10)
    fitness_values = [[] for _ in range(len(decay_rates))]

    for i in range(len(decay_rates)):
        decay_rates[i] = int(decay_rates[i] * 100) / 100

    fitness_values = [[] for _ in range(len(decay_rates))]

    for i, decay_rate in enumerate(decay_rates):
        for init_temp in init_temps:
            samples = []
            for _ in range(reset):
                decay = mlrose_hiive.GeomDecay(init_temp=init_temp, decay=decay_rate)
                _, fitness_value, _ = mlrose_hiive.simulated_annealing(algo, decay)
                samples.append(fitness_value)
            fitness_values[i].append(np.mean(samples))

    param_opt_graph(
        np.array(fitness_values),
        f'Simulated Annealing tuning (GeomDecay) - '
        f'{repr(algo.fitness_fn).split(".")[-1].split(" ")[0]}',
        'Initial_Temperature',
        'Rate_Of_Decay',
        init_temps,
        decay_rates
    )
    
    init_temps = np.linspace(1.0, 10.0, 10)
    decay_rates = np.linspace(0.1, 0.99, 10)
    fitness_values = [[] for _ in range(len(decay_rates))]

    for i in range(len(decay_rates)):
        decay_rates[i] = int(decay_rates[i] * 100) / 100

    fitness_values = [[] for _ in range(len(decay_rates))]

    for i, decay_rate in enumerate(decay_rates):
        for init_temp in init_temps:
            samples = []
            for _ in range(reset):
                decay = mlrose_hiive.ArithDecay(init_temp=init_temp, decay=decay_rate)
                _, fitness_value, _ = mlrose_hiive.simulated_annealing(algo, decay)
                samples.append(fitness_value)
            fitness_values[i].append(np.mean(samples))

    param_opt_graph(
        np.array(fitness_values),
        f'Simulated Annealing tuning (ArithDecay) - '
        f'{repr(algo.fitness_fn).split(".")[-1].split(" ")[0]}',
        'Initial_Temperature',
        'Rate_Of_Decay',
        init_temps,
        decay_rates
    )

def param_tune_rhc(algo, reset=2):
    fitness_values = []
    times = []
    restarts = range(0, 5001, 1000)
    for restart in restarts:
        samples = []
        time_samples = []
        for _ in range(reset):
            start = time.time()
            _, fitness_value, _ = mlrose_hiive.random_hill_climb(algo, restarts=restart)
            time_samples.append(time.time() - start)
            samples.append(fitness_value)
        fitness_values.append(np.mean(samples))
        times.append(np.mean(time_samples))

    fitness_values = np.array(fitness_values)
    times = np.array(times)

    with plot.style.context('ggplot'):
        fig, ax1 = plot.subplots()
        plot.title(f'Effects of Varying Restarts on RHC')
        ax1.set_xlabel('Number_Of_Restarts')
        ax1.set_ylabel('Fitness')
        ax1.tick_params(axis='y')

        ax1.plot(restarts, fitness_values, 'o-', color='r')

        with plot.style.context('ggplot'):
            ax2 = ax1.twinx()

            ax2.set_ylabel('Time (in sec)')

            ax2.plot(restarts, times, color='g')
            ax2.tick_params(axis='y')

        fig.tight_layout()
        plot.grid()
        plot.show()

def param_tune_mimic(algo, reset=2):
    keep_pcts = np.linspace(0.00001, 0.1, 5)
    pop_sizes = range(100, 501, 100)

    fitness_values = [[] for _ in range(len(keep_pcts))]

    for i in range(len(keep_pcts)):
        keep_pcts[i] = int(keep_pcts[i] * 100000) / 100000

    for pop_size in pop_sizes:
        for i, keep_pct in enumerate(keep_pcts):
            samples = []
            for _ in range(reset):
                _, fitness_value, _ = mlrose_hiive.mimic(algo, keep_pct=keep_pct, pop_size=pop_size)
                samples.append(fitness_value)
            fitness_values[i].append(np.mean(samples))

    fitness_values = np.array(fitness_values)
    param_opt_graph(
        np.array(fitness_values),
        f'MIMIC tuning on '
        f'{repr(algo.fitness_fn).split(".")[-1].split(" ")[0]}',
        'Population_Size',
        'Keeping_Percentage',
        pop_sizes,
        keep_pcts
    )

def param_opt_graph(fitness_array, title, x_label, y_label, x_val, y_val):

    plot.style.use('default')
    fig, ax = plot.subplots()
    im = ax.imshow(fitness_array, interpolation='nearest', cmap='viridis')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(fitness_array.shape[1]),
           yticks=np.arange(fitness_array.shape[0]),
           xticklabels=x_val, yticklabels=y_val,
           title=title,
           ylabel=y_label,
           xlabel=x_label)

    plot.setp(ax.get_xticklabels(), rotation=30, ha="right")
    cutoff = fitness_array.max() / 2.0
    for i in range(fitness_array.shape[0]):
        for j in range(fitness_array.shape[1]):
            ax.text(j,
                    i,
                    format(int(fitness_array[i, j]), 'd'),
                    ha="center",
                    va="center",
                    color="white" if fitness_array[i, j] < cutoff else "black"
                    )
    fig.tight_layout()
    np.set_printoptions(precision=2)
    plot.show()

def Performance_Check(fitness, dict_param, lengths=range(20, 81, 10)):
    fitness_copy = deepcopy(fitness)
    fitness_name = repr(fitness).split('.')[-1].split(' ')[0]
    TheMatrix = np.array([
        Performance_Check_algorithm(fitness_copy, 'ga', dict_param, lengths),
        Performance_Check_algorithm(fitness_copy, 'sa',dict_param, lengths),
        Performance_Check_algorithm(fitness_copy, 'rhc',dict_param, lengths),
        Performance_Check_algorithm(fitness_copy, 'mimic',dict_param, lengths)])
    performance_graph(TheMatrix, lengths, fitness_name)

def Performance_Check_algorithm(fitness, algorithm, dict_param, lengths, trials=2):
    best_fitnesses = []
    exec_times = []
    fitness_name = repr(fitness).split('.')[-1].split(' ')[0]
    for length in lengths:
        problem = mlrose_hiive.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
        samples = []
        time_samples = []
        for _ in range(trials):
            start = time.time()
            if algorithm == 'ga':
                _, fitness_value, _ = mlrose_hiive.genetic_alg(problem,pop_size=dict_param['ga'][0],mutation_prob=dict_param['ga'][1])
            elif algorithm == 'sa':
                 _, fitness_value, _ = mlrose_hiive.simulated_annealing(problem,schedule=dict_param['sa'])
            elif algorithm == 'rhc':
                 _, fitness_value, _ = mlrose_hiive.random_hill_climb(problem,restarts = dict_param['rhc'])
            else:
                 _, fitness_value, _ = mlrose_hiive.mimic(problem,pop_size=dict_param['mimic'][0],keep_pct=dict_param['mimic'][1])
            time_samples.append(time.time() - start)
            samples.append(fitness_value)

        best_fitnesses.append(np.mean(samples))
        exec_times.append(np.mean(time_samples))
    best_fitnesses = np.array(best_fitnesses)
    exec_times = np.array(exec_times)
    return best_fitnesses, exec_times


def performance_graph(batches, lengths, fitness_name): 

    plot.style.use('ggplot')
    plot.title(f'{fitness_name} - Convergence Plot w.r.t 4 algorithms')
    plot.xlabel('Problem_Length')
    plot.ylabel('Fitness')

    values_batches = batches[:, 0]
    times_batches = batches[:, 1]
    values_std_batches = values_batches
    times_std_batches = times_batches
    for values, values_std in zip(values_batches, values_std_batches):
        plot.plot(lengths, values, 'o-')
    
    plot.legend(['Genetic Algorithm', 'Simulated Annealing', 'Randomized Hill Climbing (w/ Restart)', 'MIMIC'], loc='upper left')
    plot.show()

    plot.style.use('ggplot')
    plot.title(f'Execution time of each algorithm on {fitness_name}')
    plot.xlabel('Problem_Length')
    plot.ylabel('Time taken (in sec)')
    for times, times_std in zip(times_batches, times_std_batches):
        plot.plot(lengths, times, 'o-')
    plot.legend(['Genetic algorithm', 'Simulated annealing', 'Randomized hill climbing', 'MIMIC'], loc='upper left')
    plot.show()


if __name__ == "__main__":

    # FourPeaks  - GA
    Fourpeaks = mlrose_hiive.FourPeaks(t_pct=0.1)
    Fourpeaks_problem = mlrose_hiive.DiscreteOpt(length=50, maximize=True, max_val=2, fitness_fn=Fourpeaks)
    Fourpeaks_problem.set_mimic_fast_mode(True) 
    problem_name = 'Four Peaks'
    start = time.time()
    param_tune_ga(Fourpeaks_problem) 
    print('Time taken for optimizing GA: ',time.time() - start)
    start = time.time()
    param_tune_sa(Fourpeaks_problem) 
    print('Time taken for optimizing SA: ',time.time() - start)
    start = time.time()
    param_tune_rhc(Fourpeaks_problem) 
    print('Time taken for optimizing RHC: ',time.time() - start)
    start = time.time()
    param_tune_mimic(Fourpeaks_problem) 
    print('Time taken for optimizing MIMIC: ',time.time() - start)

    start = time.time()
    dict_param = {'rhc':3000, 'sa':mlrose_hiive.GeomDecay(5, 0.39), 'ga':[500, 0.5], 'mimic':[500, 0.1]} 
    Performance_Check(Fourpeaks, dict_param)
    print('FourPeaks Fitness Optimization: ',time.time() - start)
#FourPeaks Fitness Optimization:  611s

    # Flip Flop
    FF = mlrose_hiive.FlipFlop()
    FF_problem = mlrose_hiive.DiscreteOpt(length=50, maximize=True, max_val=2, fitness_fn=FF)
    FF_problem.set_mimic_fast_mode(True) 
    problem_name = 'KnapSack'
    start = time.time()
    param_tune_ga(FF_problem)
    print('Time taken for optimizing GA: ',time.time() - start)
    start = time.time()
    param_tune_sa(FF_problem) 
    print('Time taken for optimizing SA: ',time.time() - start)
    start = time.time()
    param_tune_rhc(FF_problem) 
    print('Time taken for optimizing RHC: ',time.time() - start)
    start = time.time()
    param_tune_mimic(FF_problem) 
    print('Time taken for optimizing MIMIC: ',time.time() - start)
    #Performance_Check(Conpeaks, dict_param)  #
#Time taken for optimizing GA:  113s
#Time taken for optimizing SA:  26s
#Time taken for optimizing RHC:  14114s
#Time taken for optimizing MIMIC:  72s
    start = time.time()
    dict_param = {'rhc':3000, 'sa':mlrose_hiive.GeomDecay(4, 0.99), 'ga':[400, 0.05], 'mimic':[500, 0.1]} #0.025mimic, mlrose_hiive.ExpDecay(8, 0.0001)
    Performance_Check(FF, dict_param)
    print('FlipFlop Fitness Optimization: ',time.time() - start)


    # ContinuousPeaks Color - SA
    Conpeaks = mlrose_hiive.ContinuousPeaks(t_pct=0.1)
    Conpeaks_problem = mlrose_hiive.DiscreteOpt(length=50, maximize=True, max_val=2, fitness_fn=Conpeaks)
    Conpeaks_problem.set_mimic_fast_mode(True) 
    problem_name = 'Continuous Peaks'
    start = time.time()
    param_tune_ga(Conpeaks_problem)  
    print('Time taken for optimizing GA: ',time.time() - start)
    start = time.time()
    param_tune_sa(Conpeaks_problem) 
    print('Time taken for optimizing SA: ',time.time() - start)
    start = time.time()
    param_tune_rhc(Conpeaks_problem) 
    print('Time taken for optimizing RHC: ',time.time() - start)
    start = time.time()
    param_tune_mimic(Conpeaks_problem) 
    print('Time taken for optimizing Mimic: ',time.time() - start)
#Time taken for optimizing GA:  83s
#Time taken for optimizing SA:  25s
#Time taken for optimizing RHC:  182s
#Time taken for optimizing Mimic:  39s
    start = time.time()
    dict_param = {'rhc':3000, 'sa':mlrose_hiive.GeomDecay(8, 0.89), 'ga':[300, 0.005], 'mimic':[500, 0.1]}
    Performance_Check(Conpeaks, dict_param)
    print('ContinuousPeaks Fitness Optimization: ',time.time() - start)
#ContinuousPeaks Fitness Optimization:  625s

"""
    # SixPeaks  - GA
    Sixpeaks = mlrose_hiive.SixPeaks(t_pct=0.1)
    Sixpeaks_problem = mlrose_hiive.DiscreteOpt(length=50, maximize=True, max_val=2, fitness_fn=Sixpeaks)
    Sixpeaks_problem.set_mimic_fast_mode(True) 
    problem_name = 'Six Peaks'
    start = time.time()
    param_tune_ga(Sixpeaks_problem)  # ???
    print('Time taken for optimizing GA: ',time.time() - start)
    start = time.time()
    param_tune_sa(Sixpeaks_problem)  # ???
    print('Time taken for optimizing SA: ',time.time() - start)
    start = time.time()
    param_tune_rhc(Sixpeaks_problem)  # ???
    print('Time taken for optimizing RHC: ',time.time() - start)
    start = time.time()
    param_tune_mimic(Sixpeaks_problem)  # ???
    print('Time taken for optimizing MIMIC: ',time.time() - start)
#Time taken for optimizing GA:  61.60065174102783
#Time taken for optimizing SA:  221.82194662094116
#Time taken for optimizing RHC:  203.39129376411438
#Time taken for optimizing MIMIC:  31.211236715316772
    start = time.time()
    dict_param = {'rhc':4000, 'sa':mlrose_hiive.GeomDecay(5, 0.1), 'ga':[300, 0.05], 'mimic':[500, 0.1]}
    Performance_Check(Sixpeaks, dict_param)
    print('SixPeaks Fitness Optimization: ',time.time() - start)
"""