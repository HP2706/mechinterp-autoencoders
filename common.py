from modal import Stub, Volume
stub = Stub(name="autoencoder anthropic")
vol = Volume.from_name("autoencoder anthropic", create_if_missing=True)
PATH = "/autoencoder"