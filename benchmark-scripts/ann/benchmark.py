from functions import *


if __name__ == '__main__':

    # variables
    weaviate_urls = ['http://weaviate1:8080', 'http://weaviate2:8080', 'http://weaviate3:8080']
    CPUs = 16
    efConstruction_array = [64, 128]
    maxConnections_array = [16, 32]
    ef_array = [64, 128, 256, 512]

    benchmark_params_array = [
        {
            'dataset_folder': 'dataset_100_000',
            'distance_metric': 'cosine',
            'multinode': True,
            'replication_factor': 3,
            'shards': 1,
        }
    ]

    # Starts the actual benchmark, prints "completed" when done
    run_the_benchmarks(weaviate_urls, CPUs, efConstruction_array, maxConnections_array, ef_array, benchmark_params_array)
