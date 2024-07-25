from modal_common import app, image
from modal import gpu, Mount

dir = Mount.from_local_dir(".", remote_path="/root")

@app.function(
    image=image,
    gpu=gpu.A10G(),
    mounts=[dir],
    _allow_background_volume_commits=True
)
def test():
    import subprocess
    subprocess.run(["pytest", "perf.py"], check=True, cwd="/root/benchmarks")

@app.local_entrypoint()
def main():
    test.remote()