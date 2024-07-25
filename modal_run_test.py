from modal import Mount, gpu
from modal_common import app, image

#mount the tests directory
tests = Mount.from_local_dir(".", remote_path="/root/tests")

#run all tests on modal
# this enables us to run tests on a GPU 
@app.function(
    gpu=gpu.A10G(count=1),  
    image=image,
    mounts=[tests]
)
def run_tests():
    import subprocess
    subprocess.run(["pytest", "tests"], check=True, cwd="/root")

@app.local_entrypoint()
def main():
    run_tests.remote()
