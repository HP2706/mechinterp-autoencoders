from modal import Image, App, Volume
app = App("test-autoencoders")

vol = Volume.from_name("benchmark-autoencoders", create_if_missing=True)

image = Image.debian_slim(python_version="3.11").run_commands(
    'apt update'
).pip_install(
    'uv', 
).env(
    {"VIRTUAL_ENV": "/usr/local",
    "PATH" : "/usr/local/bin:$PATH"}
).copy_local_file(
    'requirements.txt', '/root/requirements.txt'
).run_commands(
    'uv pip install -r /root/requirements.txt'
)

