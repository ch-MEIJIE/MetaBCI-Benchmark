from benchopt import run_benchmark

run_benchmark(
    benchmark_path='.',
    solver_names=[
        'MetaBCI-docomposition-algo',
    ],
    dataset_names=[
        'Wang2016[channel=occipital_9,duration=0.8,subject=[1,2]]',
    ]
)
