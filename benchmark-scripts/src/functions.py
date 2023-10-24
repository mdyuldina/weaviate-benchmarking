import os
import uuid
import json
import time
import subprocess
import h5py
import weaviate
from loguru import logger
import multiprocessing as mp

BENCHMARK_IMPORT_BATCH_SIZE = 100


def handle_results(results):
    """Handle error message from batch requests
       logs the message as an info message."""
    if results is not None:
        for result in results:
            if 'result' in result and 'errors' in result['result'] and 'error' in result['result']['errors']:
                for message in result['result']['errors']['error']:
                    logger.error(message['message'])


def match_results(test_set, weaviate_result_set, k):
    """Match the results from Weaviate to the benchmark data.
       If a result is in the returned set, score goes +1.
       Because there is checked for 100 neighbors a score
       of 100 == perfect"""

    # set score
    score = 0

    # return if no result
    if not weaviate_result_set['data']['Get']['Benchmark']:
        return score

    # create array from Weaviate result
    weaviate_result_array = []
    for weaviate_result in weaviate_result_set['data']['Get']['Benchmark'][:k]:
        weaviate_result_array.append(weaviate_result['counter'])

    # match scores
    for nn in test_set[:k]:
        if nn in weaviate_result_array:
            score += 1
    
    return score


def run_speed_test(l, CPUs, weaviate_url):
    """Runs the actual speed test in Go"""
    process = subprocess.Popen(
        ['./benchmarker', 'dataset', '-u', weaviate_url, '-c', 'Benchmark', '-q', 'queries.json', '-p', str(CPUs), '-f',
         'json', '-l', str(l)], stdout=subprocess.PIPE)
    result_raw = process.communicate()[0].decode('utf-8')
    try:
        result = json.loads(result_raw)
    except:
        logger.error('Faulty response:')
        logger.error(result_raw)
        result = {}
    return result


def do_queries(queue, k, weaviate_url, times, processed):
    while True:
        job = queue.get()
        if not job:
            break
        client = weaviate.Client(weaviate_url, timeout_config=(5, 60))
        for query in job:
            nearVector = {"vector": query}
            start = time.time()
            query_result = client.query.get("Benchmark", ["counter"]).with_near_vector(nearVector).with_limit(k).do()
            stop = time.time()
            if query_result['data'] and 'errors' not in query_result:
                with times.get_lock():
                    times.value += (stop - start)
                with processed.get_lock():
                    processed.value += 1
        queue.task_done()
    queue.task_done()


def run_speed_test_python(k, CPUs, weaviate_urls):
    """Runs speed test with python client, about 10 times slower than in Go"""
    logger.info(f'Run speed test for k={k}')
    processed = mp.Value('i', 0)
    times = mp.Value('f', 0)
    with open('queries.json', 'r', encoding='utf-8') as jf:
        queries = json.load(jf)
    queue = mp.JoinableQueue()
    for _ in range(CPUs):
        queue.put(queries)
    for i in range(CPUs):
        queue.put(None)

    workers = []
    for i in range(CPUs):
        n_url = i % len(weaviate_urls)
        worker = mp.Process(target=do_queries,
                            args=(queue, k, weaviate_urls[n_url], times, processed))
        workers.append(worker)
        worker.start()
    queue.join()
    qps = processed.value / times.value
    logger.info(f'qps for k={k}: {qps}')
    return processed.value / times.value


def conduct_benchmark(weaviate_urls, CPUs, ef, client, benchmark_params, efConstruction, maxConnections):
    """Conducts the benchmark, note that the NN results
       and speed test run seperatly from each other"""

    # result obj
    results = {
        'benchmarkFolder': benchmark_params['dataset_folder'],
        'distanceMetric': benchmark_params['distance_metric'],
        'totalTested': 0,
        'ef': ef,
        'efConstruction': efConstruction,
        'maxConnections': maxConnections,
        'recall': {
            '100': {
                'highest': 0,
                'lowest': 100,
                'average': 0
            },
            '10': {
                'highest': 0,
                'lowest': 100,
                'average': 0
            },
            '1': {
                'highest': 0,
                'lowest': 100,
                'average': 0
            },
        },
        'requestTimes': {}
    }

    # update schema for ef setting
    logger.info('Update "ef" to ' + str(ef) + ' in schema')
    client.schema.update_config('Benchmark', {'vectorIndexConfig': {'ef': ef}})

    ##
    # Run the score test
    ##
    all_scores = {
        '100': [],
        '10': [],
        '1': [],
    }

    logger.info('Find neighbors with ef = ' + str(ef))
    c = 0
    with h5py.File('/var/lib/benchmark/hdf5/' + benchmark_params['dataset_folder'] + '/test.hdf5', 'r') as test_file:
        with h5py.File('/var/lib/benchmark/hdf5/' + benchmark_params['dataset_folder'] + '/neighbors.hdf5',
                       'r') as neighbors_file:
            test_vectors = test_file['test']
            test_vectors_len = len(test_vectors)

            for test_vector in test_vectors:

                # set certainty for  l2-squared
                nearVector = {"vector": test_vector.tolist()}

                # Start request
                query_result = client.query.get("Benchmark", ["counter"]).with_near_vector(nearVector).with_limit(100).do()

                for k in [1, 10, 100]:
                    k_label = f'{k}'
                    score = match_results(neighbors_file['neighbors'][c], query_result, k)
                    if k > 10 and score == 0:
                        logger.info(
                            'There is a 0 score, this most likely means there is an issue with the dataset OR you have very low index settings')
                    all_scores[k_label].append(score)

                    # set if high and low score
                    if score > results['recall'][k_label]['highest']:
                        results['recall'][k_label]['highest'] = score
                    if score < results['recall'][k_label]['lowest']:
                        results['recall'][k_label]['lowest'] = score

                # log ouput
                if (c % 1000) == 0:
                    logger.info('Validated ' + str(c) + ' of ' + str(test_vectors_len))

                c += 1

    ##
    # Run the speed test
    ##
    logger.info('Run the speed test')
    with h5py.File('/var/lib/benchmark/hdf5/' + benchmark_params['dataset_folder'] + '/train.hdf5', 'r') as f:
        train_vectors_len = len(f['train'])
    with h5py.File('/var/lib/benchmark/hdf5/' + benchmark_params['dataset_folder'] + '/test.hdf5', 'r') as f:
        vector_write_array = []
        for vector in f['test']:
            vector_write_array.append(vector.tolist())
        with open('queries.json', 'w', encoding='utf-8') as jf:
            json.dump(vector_write_array, jf, indent=2)

        if benchmark_params['multinode']:
            results['requestTimes']['limit_1'] = {'qps': run_speed_test_python(1, CPUs, weaviate_urls)}
            results['requestTimes']['limit_10'] = {'qps': run_speed_test_python(10, CPUs, weaviate_urls)}
            results['requestTimes']['limit_100'] = {'qps': run_speed_test_python(100, CPUs, weaviate_urls)}
        else:
            results['requestTimes']['limit_1'] = run_speed_test(1, CPUs, weaviate_urls[0])
            results['requestTimes']['limit_10'] = run_speed_test(10, CPUs, weaviate_urls[0])
            results['requestTimes']['limit_100'] = run_speed_test(100, CPUs, weaviate_urls[0])

    # add final results
    results['totalTested'] = c
    results['totalDatasetSize'] = train_vectors_len
    for k in ['1', '10', '100']:
        results['recall'][k]['average'] = sum(all_scores[k]) / len(all_scores[k])

    return results


def remove_weaviate_class(client):
    """Removes the main class and tries again on error"""
    try:
        client.schema.delete_all()
        # Sleeping to avoid load timeouts
    except:
        logger.exception('Something is wrong with removing the class, sleep and try again')
        time.sleep(240)
        remove_weaviate_class(client)


def import_into_weaviate(client, efConstruction, maxConnections, benchmark_params):
    """Imports the data into Weaviate"""

    # variables
    benchmark_class = 'Benchmark'

    # Delete schema
    remove_weaviate_class(client)

    # Create schema
    schema = {
        "classes": [{
            "class": benchmark_class,
            "description": "A class for benchmarking purposes",
            "properties": [
                {
                    "dataType": [
                        "int"
                    ],
                    "description": "The number of the counter in the dataset",
                    "name": "counter"
                }
            ],
            "vectorIndexConfig": {
                "ef": -1,
                "efConstruction": efConstruction,
                "maxConnections": maxConnections,
                "vectorCacheMaxObjects": 1000000000,
                "distance": benchmark_params['distance_metric']
            }
        }]
    }
    multinode = benchmark_params['multinode']
    if multinode:
        schema['classes'][0]['replicationConfig'] = {"factor": benchmark_params['replication_factor']}
        shards = benchmark_params['shards']
        schema['classes'][0]['shardingConfig'] = {"desiredCount": shards}
        schema['classes'][0]['shardingConfig'] = {"actualCount": shards}
    client.schema.create(schema)

    # Import
    logger.info('Start import process for ' + benchmark_params['dataset_folder'] + ', ef' + str(
        efConstruction) + ', maxConnections' + str(maxConnections))
    with h5py.File('/var/lib/benchmark/hdf5/' + benchmark_params['dataset_folder'] + '/train.hdf5', 'r') as f:
        vectors = f['train']
        vector_len = len(vectors)
        start_import_time = time.time()
        with client.batch as batch:
            for i, vector in enumerate(vectors):
                batch.add_data_object({
                    'counter': i
                },
                    'Benchmark',
                    str(uuid.uuid3(uuid.NAMESPACE_DNS, str(i))),
                    vector=vector
                )
                if i % 10000 == 0:
                    logger.info(f'Imported {i} objects from {vector_len} which is {i * 100 // vector_len}%')
                    nodes_status = client.cluster.get_nodes_status()
                    for node in nodes_status:
                        name = node['name']
                        objects = node['stats']['objectCount']
                        logger.info(f'node {name} has {objects} objects')
        stop_import_time = time.time()
        nodes_status = client.cluster.get_nodes_status()
        for node in nodes_status:
            name = node['name']
            objects = node['stats']['objectCount']
            logger.info(f'node {name} has {objects} objects')
        import_time = stop_import_time - start_import_time
        logger.info(f'done importing {vector_len} objects in {import_time} seconds')

    return import_time


def run_the_benchmarks(weaviate_urls, CPUs, efConstruction_array, maxConnections_array, ef_array,
                       benchmark_params_array):
    """Runs the actual benchmark.
       Results are stored in a JSON file"""

    # Connect to Weaviate Weaviate
    try:
        client = weaviate.Client(weaviate_urls[0], timeout_config=(5, 60))
    except:
        print('Error, can\'t connect to Weaviate, is it running?')
        exit(1)

    client.batch.configure(
        timeout_retries=10,
        batch_size=BENCHMARK_IMPORT_BATCH_SIZE,
        num_workers=8,
        dynamic=True
    )

    # iterate over settings
    for benchmark_params in benchmark_params_array:
        multinode = benchmark_params['multinode']
        for efConstruction in efConstruction_array:
            for maxConnections in maxConnections_array:
               
                # import data
                import_time = import_into_weaviate(client, efConstruction, maxConnections, benchmark_params)

                # Find neighbors based on UUID and ef settings
                results = []
                for ef in ef_array:
                    result = conduct_benchmark(weaviate_urls, CPUs, ef, client, benchmark_params, efConstruction,
                                               maxConnections)
                    result['importTime'] = import_time
                    results.append(result)

                # write json file
                if not os.path.exists('results'):
                    os.makedirs('results')
                folder = benchmark_params["dataset_folder"]
                if multinode:
                    rf = benchmark_params['replication_factor']
                    sh = benchmark_params['shards']
                    output_json = f'results/{folder}__{efConstruction}__{maxConnections}__rf_{rf}__sh_{sh}.json'
                else:
                    output_json = f'results/{folder}__{efConstruction}__{maxConnections}.json'
                logger.info('Writing JSON file with results to: ' + output_json)
                with open(output_json, 'w') as outfile:
                    json.dump(results, outfile)

    logger.info('completed')
