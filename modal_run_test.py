from modal import Mount, gpu
from modal_common import app, image

#mount the tests directory
tests = Mount.from_local_dir(".", remote_path="/root/project")

#run all tests on modal
# this enables us to run tests on a GPU 
@app.function(
    gpu=gpu.A10G(count=1),  
    image=image,
    mounts=[tests]
)
def run_tests():
    import subprocess
    import os
    import sys
    # Add the project directory to the Python path
    project_dir = "/root/project"
    sys.path.append(project_dir)
    # Change to the project directory
    os.chdir(project_dir)

    print("in run_tests os.listdir()", os.listdir())
    print("in run_tests os.getcwd()", os.getcwd())

    # Run the tests
    subprocess.run(["pytest", "tests"], check=True)


@app.local_entrypoint()
def main():
    run_tests.remote()
