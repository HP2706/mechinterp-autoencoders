from modal_common import app, image, vol
from modal import gpu, Mount
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.run_benchmark import test

@app.function(
    image=image,
    volumes={"/root/modal_benchmark": vol},
    gpu=gpu.A100(),
    timeout=60*60, #1 hours
    _allow_background_volume_commits=True
)
def benchmark(cleanup: bool = False):  
    if cleanup and os.path.exists("/root/modal_benchmark/data"):
        print("removing /root/modal_benchmark/data")
        os.system("rm -rf /root/modal_benchmark/data")

    os.makedirs("/root/modal_benchmark/data", exist_ok=True)
    test()
    vol.commit()

@app.local_entrypoint()
def main():
    benchmark.remote(cleanup=True)
    local_dir_path = 'benchmarks/data'
    os.makedirs(local_dir_path, exist_ok=True)
    #import subprocess
    #subprocess.run(["modal", "volume", "get", "benchmark-autoencoders", "benchmark_autoencoders/data", "--force"], check=True)
